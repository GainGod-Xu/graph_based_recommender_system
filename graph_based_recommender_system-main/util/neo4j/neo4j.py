from py2neo import Graph


class Neo4j:
    def __init__(self, uri, user, password):
        self.graph = Graph(uri, auth=(user, password))

    def query(self, query):

        # TODO: implement neo4j query
        # 1. insert nodes
        # 2. insert/define relations
        # 3. fetch stuff
        pass

    # def insert(self, )
