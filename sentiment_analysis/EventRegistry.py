from eventregistry import *
import datetime

x = datetime.datetime(2018, 3, 1)
er = EventRegistry(apiKey="24fd488a-afd4-4daa-abe3-cd4e484e5d30")
q = QueryArticlesIter(categoryUri=er.getCategoryUri("Healthcare"),
                      sourceLocationUri=er.getLocationUri("USA"),
                      locationUri=er.getLocationUri("USA"),
                      lang="eng",
                      dataType=["news","pr"],
                      dateEnd=x)
for art in q.execQuery(er, sortBy="date"):
    print(art)
