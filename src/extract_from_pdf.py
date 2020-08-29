#!/usr/bin/env python
import tika
import re
from tika import parser


def extract_data_from_file(file_path):
    parsed = parser.from_file(file_path)
    # print(parsed["metadata"])
    content = parsed["content"]
    # change all matchObj to a more meaningful name for each
    # then make the list at the end of this
    # print(content)
    re_date = re.search(r'Actual data of ([\d\.]+)', content)
    # print("Actual data of ", matchObj.group(1))
    # data.append(matchObj.group(1))

    re_max_demand = re.search(r'Max\. Demand \(Generation end\) : ([\d\.]+)', content)
    # print("Max Demand (Generation end", matchObj.group(1))
    # data.append(matchObj.group(1))

    re_max_demand_sub = re.search(r'Max\. Demand \(Sub-station end\) : ([\d\.]+)', content)
    # print('Max.Demand(Sub station end)', matchObj.group(1))
    # data.append(matchObj.group(1))

    re_highest_gen = re.search(r'Highest Generation \(Generation end\) : ([\d\.]+)', content)
    # print('Highest Generation (Generation end) :', matchObj.group(1))

    re_total_gas = re.search(r'Total Gas Supplied : ([\d\.]+)', content)
    # print("Total Gas", matchObj.group(1))

    re_water_level = re.search(r'Water Level of Kaptai Lake at 06:00 AM Yesterday = ([\d\.]+)', content)
    # print("Water Level", matchObj.group(1))

    re_rule_cuve = re.search(r'Rule Curve = ([\d\.]+)', content)
    # print("Rule curve", matchObj.group(1))

    re_total_energy = re.search(r'Total Energy \(Generation \+ India Import\) : ([\d\.]+)', content)
    # print("Total Energy", matchObj.group(1))

    re_dhaka_temp = re.search(r'Temperature in Dhaka was : ([\d\.]+)', content)
    # print("Temperature", matchObj.group(1))

    re_min_gen_and_dhaka_mym = re.search(
        r'Minimum Generation \(Generation end\) : ([\d+\.]+) MW,   at = [\d+\:\.]+(  hrs)? Dhaka (\d+) \d+ \d+ Mymensingh (\d+)',
        content)
    # print("Dhaka", matchObj.group(2), "Mymensingh", matchObj.group(3))

    # print("Minimum generation", matchObj.group(1)) Dhaka

    re_day_peak_and_chitt_syl = re.search(
        r'Day-peak Generation \(Generation end\) : ([\d\.]+) MW,   at = [\d\:\.]+  hrs (Chattogram|Chittagong) (\d+) \d+ \d+ Sylhet (\d+)',
        content)
    # print("Chattogram", matchObj.group(3), "Sylhet", matchObj.group(4))

    # print("Day peak gen", matchObj.group(1))

    re_evening_peak_and_khulna_barishal = re.search(
        r'Evening-peak Generation \(Generation end\) : ([\d\.]+) MW,   at = [\d\:\.]+  hrs Khulna (\d+) [\d]+ \d+ (Barishal|Barisal) (\d+)',
        content)
    # print("Khulna", matchObj.group(2), "Barisal", matchObj.group(4))

    # print("Evening peak", matchObj.group(1))

    re_evening_load_and_raj_rang = re.search(
        r'Evening Peak Load-shed \(Sub-station end\) : ([\d\.]+) MW,   at = [\d\:\.]+  hrs Rajshahi (\d+) [\d]+ \d+ Rangpur (\d+)',
        content)
    # print("Rajshahi", matchObj.group(2), "Rangpur", matchObj.group(3))

    # print("Evening shed", matchObj.group(1))

    # re_cumilla = re.search(r'Generation shortfall at evening peak due to : (Cumilla|Comilla) (\d+)', content)
    # print("Cumilla", matchObj.group(2))
    data = []
    data.append(re_date.group(1))
    data.append(re_max_demand.group(1))
    data.append(re_max_demand_sub.group(1))
    data.append(re_highest_gen.group(1))
    data.append(re_min_gen_and_dhaka_mym.group(1))
    data.append(re_day_peak_and_chitt_syl.group(1))
    data.append(re_evening_peak_and_khulna_barishal.group(1))
    data.append(re_evening_load_and_raj_rang.group(1))
    data.append(re_water_level.group(1))
    data.append(re_rule_cuve.group(1))
    data.append(re_dhaka_temp.group(1))
    data.append(re_total_gas.group(1))
    data.append(re_total_energy.group(1))
    data.append(re_min_gen_and_dhaka_mym.group(3))
    data.append(re_min_gen_and_dhaka_mym.group(4))
    data.append(re_day_peak_and_chitt_syl.group(3))
    data.append(re_day_peak_and_chitt_syl.group(4))
    data.append(re_evening_peak_and_khulna_barishal.group(2))
    data.append(re_evening_peak_and_khulna_barishal.group(4))
    data.append(re_evening_load_and_raj_rang.group(2))
    data.append(re_evening_load_and_raj_rang.group(3))
    # data.append(re_cumilla.group(2))

    return data


if __name__ == "__main__":
    data = extract_data_from_file("../pdf/2780report.pdf")
    print(data)
