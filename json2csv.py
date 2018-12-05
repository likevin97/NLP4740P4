import sys, json
with open(sys.argv[1]) as f:
    predictions = json.load(f)

with open(sys.argv[2] + '.csv', 'w') as f:
    f.write("Id,Category\n")
    for k,v in predictions.items():
        f.write("%s,%s\n"%(k,v))

    f.close()