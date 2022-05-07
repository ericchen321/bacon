# filter out not-needed packages from pip-generated
# requirements.txt

import subprocess
from re import search

with open('requirements_raw.txt','r') as raw_req_file:
    with open('requirements_filtered.txt','w') as filtered_req_file:
        dep_lines = raw_req_file.readlines()
        for dep_line in dep_lines:
            if "==" in dep_line or "@" in dep_line:
                if "==" in dep_line:
                    package_name = dep_line.split("==")[0]
                else:
                    package_name = dep_line.split(" @ ")[0]
                out = subprocess.check_output(
                    f"grep -r -n -I '{package_name}' ../", stderr=subprocess.STDOUT, shell=True).decode('utf-8')
                if search("\.py", out):
                    print(f"package {package_name} used")
                    filtered_req_file.write(dep_line)
            