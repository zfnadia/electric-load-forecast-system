from pdf_extract import extract_data_from_file

header = [
    "Date",
    "Max. Demand (Generation end) MW",
    "Max. Demand (Sub-station end) MW",
    "Highest Generation (Generation end) MW",
    "Minimum Generation (Generation end) MW",
    "Day-peak Generation (Generation end) MW",
    "Evening-peak Generation (Generation end) MW",
    "Evening Peak Load-shed (Sub-station end) MW",
    "Water Level of Kaptai Lake at 06:00 AM ft",
    "Rule Curve ft",
    "Maximum Temperature in Dhaka ° C",
    "Total Gas Supplied MMCFD",
    "Total Energy (Generation + India Import) MKWh",
    "Dhaka (Demand at Evening Peak)",
    "Mymensingh (Demand at Evening Peak)",
    "Chattogram (Demand at Evening Peak)",
    "Sylhet (Demand at Evening Peak)",
    "Khulna (Demand at Evening Peak)",
    "Barishal (Demand at Evening Peak)",
    "Rajshahi (Demand at Evening Peak)",
    "Rangpur (Demand at Evening Peak)",
    # "Cumilla (Demand at Evening Peak)",
]


def generate_csv():
    start = 4198
    end = 4198
    with open("../csv_files/output.csv", "w") as csv:
        csv.write(",".join(header))
        csv.write("\n")
        while start <= end:
            try:
                print("Processing: ", start)
                row = extract_data_from_file("../pdf/%sreport.pdf" % start)
                csv.write(",".join(row))
                csv.write("\n")
            except FileNotFoundError as e:
                pass
            start += 1


if __name__ == "__main__":
    generate_csv()
