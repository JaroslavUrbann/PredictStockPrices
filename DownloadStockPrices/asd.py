import csv
path = 'C:\\Users\\JaroslavUrban\\Desktop\\StockPricePredicting\\Companies2\\najs.csv'
with open(path, "w", newline="") as newfile:
    csv_writer = csv.writer(newfile)