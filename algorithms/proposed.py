import numpy as np
import pandas as pd
from surprise import reader
from surprise.prediction_algorithms import SlopeOne
import skfuzzy as fuzz


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

def assign_class(row):
    if row["r_low"] > row["r_mid"] and row["r_low"] > row["r_high"]:
        return "low"
    elif row["r_mid"] > row["r_low"] and row["r_mid"] > row["r_high"]:
        return "mid"
    elif row["r_high"] > row["r_low"] and row["r_high"] > row["r_mid"]:
        return "high"
    return "var"


def remove_noise_proposed(df,dataset_name="data"):
    min_rating = df["rating"].min()
    max_rating = df["rating"].max()
    reader = reader.Reader(rating_scale=(min_rating, max_rating))
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
    df_user = df[["user", "r_low", "r_mid", "r_high"]].groupby("user").mean()
    df_user["cat"] = df_user.apply(assign_class, axis=1)

    df_item = df[["item", "r_low", "r_mid", "r_high"]].groupby("item").mean()
    df_item["cat"] = df_item.apply(assign_class, axis=1)

    new_ratings = []


    df["rating_cat"] = df.apply(assign_class,axis=1)
