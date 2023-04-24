import skfuzzy as fuzz
import numpy as np
import sys
from surprise import Dataset
from surprise import SlopeOne, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
from surprise.reader import Reader
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from multiprocessing import Pool
import pathlib


def get_accuracy(algo, trainset, testset):
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse_error = rmse(predictions)
    mae_error = mae(predictions)
    print(f"For {algo} RMSE is {rmse_error} and MAE is {mae_error}")
    return algo, rmse_error, mae_error

def run_algos(n_jobs,trainset,testset):
    algorithms = [
        SlopeOne(),
        KNNBasic(k=60, sim_options={"name": "pearson", "user_based": True}),
        KNNBasic(k=60, sim_options={"name": "pearson", "user_based": False}),
    ]

    if n_jobs == 1:
        results = []
        for algo in algorithms:
            results.append(get_accuracy(algo, trainset, testset))
    else:
        args = [(algo, trainset, testset) for algo in algorithms]
        with Pool(n_jobs) as p:
            results = p.starmap(get_accuracy, args)
    return results

def main():
    if len(sys.argv) < 2:
        fname = input("Enter the file name: ")
    else:
        fname = sys.argv[1]
    if len(sys.argv) == 3:
        max_rating = int(sys.argv[2])
    else:
        max_rating = 5
    if len(sys.argv)==4:
        n_jobs = int(sys.argv[3])
    else:
        n_jobs = 1

    dataset_name = pathlib.Path(fname).stem
    df = pd.read_csv(fname)
    print(f"Dataset: {dataset_name} with rating scale (1, {max_rating})")
    print(f"Shape of the data: {df.shape}")
    reader = Reader(rating_scale=(1, max_rating))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    print("Training set size: ", trainset.n_ratings)
    print("Test set size: ", len(df) - trainset.n_ratings)

    print("Base algorithms")
    results_base = run_algos(n_jobs,trainset,testset)
    algo_base = results_base[0][0]
    results_dict = {"algo": [], "rmse": [], "mae": [],"type":[]}
    for algo, rmse, mae in results_base:
        results_dict["algo"].append(repr(algo).split(".")[-1].split(" ")[0])
        results_dict["rmse"].append(rmse)
        results_dict["mae"].append(mae)
        results_dict["type"].append("base")

    print("Correcting Noise using fuzzy")

    rating = np.arange(1, max_rating + 1, 1)

    diff = (max_rating - 1) / 2

    rating_low = fuzz.membership.gaussmf(rating, 1, diff)
    rating_mid = fuzz.membership.gaussmf(rating, 1 + diff, diff)
    rating_high = fuzz.membership.gaussmf(rating, 1 + 2 * diff, diff)
    categories = {"low": rating_low, "mid": rating_mid, "high": rating_high}

    x = np.arange(1, max_rating, 0.01)

    plt.plot(x, fuzz.interp_membership(rating, rating_low, x), label="low", color="red")
    plt.plot(
        x, fuzz.interp_membership(rating, rating_mid, x), label="mid", color="orange"
    )
    plt.plot(
        x, fuzz.interp_membership(rating, rating_high, x), label="high", color="green"
    )

    plt.ylabel("Membership")
    plt.ylabel("Rating values")
    plt.legend()

    plt.savefig(f"results/membership_{dataset_name}.png")

    print("Membership functions plotted and saved as membership.png")

    df["r_low"] = df["rating"].apply(
        lambda x: fuzz.interp_membership(rating, rating_low, x)
    )
    df["r_mid"] = df["rating"].apply(
        lambda x: fuzz.interp_membership(rating, rating_mid, x)
    )
    df["r_high"] = df["rating"].apply(
        lambda x: fuzz.interp_membership(rating, rating_high, x)
    )

    def assign_class(row):
        if row["r_low"] > row["r_mid"] and row["r_low"] > row["r_high"]:
            return "low"
        elif row["r_mid"] > row["r_low"] and row["r_mid"] > row["r_high"]:
            return "mid"
        elif row["r_high"] > row["r_low"] and row["r_high"] > row["r_mid"]:
            return "high"
        return "var"

    df_user = df[["user", "r_low", "r_mid", "r_high"]].groupby("user").mean()
    df_user["cat"] = df_user.apply(assign_class, axis=1)

    df_item = df[["item", "r_low", "r_mid", "r_high"]].groupby("item").mean()
    df_item["cat"] = df_item.apply(assign_class, axis=1)

    new_ratings = []


    df["rating_cat"] = df.apply(assign_class,axis=1)

    def get_new_rating(row):
        user = row["user"]
        item = row["item"]
        rating_old = row["rating"]
        rating_cat = row["rating_cat"]
        user_cat = df_user.loc[user, "cat"]
        item_cat = df_item.loc[item, "cat"]
        rating_new = rating_old
        if (
            (
                (user_cat == item_cat)
                and (user_cat != rating_cat)
                and (item_cat != rating_cat)
            )
            or ((user_cat == "low") and (item_cat == "mid") and (rating_cat == "high"))
            or ((user_cat == "mid") and (item_cat == "low") and (rating_cat == "high"))
            or ((user_cat == "high") and (item_cat == "mid") and (rating_cat == "low"))
            or ((user_cat == "mid") and (item_cat == "high") and (rating_cat == "low"))
        ):
            try:
                rating_new = algo_base.predict(user, item, rating).est
            except:
                rating_new = rating_old
        return rating_new

    df["new_ratings"] = df.apply(get_new_rating, axis=1)
    df.to_csv(f"results/new_ratings_{dataset_name}.csv", index=False)
    print("New ratings saved as new_ratings.csv")

    print("Calculating with new ratings")
    data_new = Dataset.load_from_df(df[["user", "item", "new_ratings"]], reader)
    trainset, testset = train_test_split(data_new, test_size=0.2, random_state=42)
    print("Training set size: ", trainset.n_ratings)
    print("Test set size: ", len(df) - trainset.n_ratings)

    results = run_algos(n_jobs,trainset,testset)
    for algo, rmse, mae in results:
        results_dict["algo"].append(repr(algo).split(".")[-1].split(" ")[0])
        results_dict["rmse"].append(rmse)
        results_dict["mae"].append(mae)
        results_dict["type"].append("fuzzy")
    df_results = pd.DataFrame(results_dict)
    df_results.to_csv(f"results/results_{dataset_name}.csv", index=False)
    print("Results saved as results.csv")


if __name__ == "__main__":
    main()
