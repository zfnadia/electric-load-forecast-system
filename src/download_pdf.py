import urllib.request

url = "http://www.bpdb.gov.bd/bpdb/pdb_utility/maxgen/dailygen_archive/%sreport.pdf"

start = 3921
end = 4203

# missing = [2799]

while start <= end:
    print("downloading %s" % start)
    try:
        urllib.request.urlretrieve(url % start, "../pdf/%sreport.pdf" % start)
    except urllib.request.HTTPError as e:
        if e.code == 404:
            # missing.append(start)
            print("Missing", start)
        else:
            raise
    else:
        print("finished %s" % start)
    start += 1

print("MISSING: ", missing)
