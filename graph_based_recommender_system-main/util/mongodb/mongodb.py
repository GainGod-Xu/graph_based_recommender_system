import json

from pymongo import MongoClient


class Data(dict):
    def __init__(self, data):
        self.process_data(data)
        self.validate_data()

    def process_data(self, data):
        pass

    def validate_data(self):
        pass

    def to_json(self):
        return json.dumps(self)


class Review(Data):
    def __init__(self, data):
        super().__init__(data)

    def process_data(self, data):
        if "overall" in data:
            self["overall"] = data["overall"]

        if "verified" in data:
            self["verified"] = data["verified"]

        if "reviewTime" in data:
            self["reviewTime"] = data["reviewTime"]

        if "reviewerID" in data:
            self["reviewerID"] = data["reviewerID"]

        if "asin" in data:
            self["asin"] = data["asin"]

        if "reviewerName" in data:
            self["reviewerName"] = data["reviewerName"]

        if "reviewText" in data:
            self["reviewText"] = data["reviewText"]

        if "summary" in data:
            self["summary"] = data["summary"]

        if "unixReviewTime" in data:
            self["unixReviewTime"] = data["unixReviewTime"]


class Meta(Data):
    def __init__(self, data):
        super().__init__(data)

    def process_data(self, data):
        if "category" in data:
            self["category"] = data["category"]

        if "tech1" in data:
            self["tech1"] = data["tech1"]

        if "description" in data:
            self["description"] = data["description"]

        if "fit" in data:
            self["fit"] = data["fit"]

        if "title" in data:
            self["title"] = data["title"]

        if "also_buy" in data:
            self["also_buy"] = data["also_buy"]

        if "image" in data:
            self["image"] = data["image"]

        if "tech2" in data:
            self["tech2"] = data["tech2"]

        if "brand" in data:
            self["brand"] = data["brand"]

        if "feature" in data:
            self["feature"] = data["feature"]

        if "rank" in data:
            self["rank"] = data["rank"]

        if "also_view" in data:
            self["also_view"] = data["also_view"]

        if "details" in data:
            self["details"] = data["details"]

        if "main_cat" in data:
            self["main_cat"] = data["main_cat"]

        if "similar_item" in data:
            self["similar_item"] = data["similar_item"]

        if "date" in data:
            self["date"] = data["date"]

        if "price" in data:
            self["price"] = data["price"]

        if "asin" in data:
            self["asin"] = data["asin"]


class Mongo:
    def __init__(
        self, host="localhost", port=27017, database="amazon_product_review", username=None, password=None
    ):
        self.client = MongoClient(host=host, port=port, username=username, password=password, authSource=database)
        self.db = self.client[database]
        self.review = self.db["review"]
        self.meta = self.db["meta"]

    def get_meta(self, query={}, limit=0):
        for item in self.meta.find(query).limit(limit):
            yield Meta(item)

    def get_review(self, query={}, limit=0):
        for item in self.review.find(query).limit(limit):
            yield Review(item)

    def get_review_by_asin(self, asin, limit=0):
        for item in self.review.find({"asin": asin}).limit(limit):
            yield Review(item)
