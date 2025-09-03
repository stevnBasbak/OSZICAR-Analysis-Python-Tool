## Dit script analyseert de oszicar. Werkt voor zowel MD als opt oszicars

############################################################################################
##                                  REQUIRED INPUTS
##
## System Definition:
oszicar = 'OSZICAR_MERGED'            # positie van de oszicar dat je wil analyseren
resolution_plots = 300         # standard 300
############################################################################################






## hier verwijderen we de folder temp_workfolder indien deze nog zou bestaan
import os
import shutil

def remove_temp_folder():
    folder_path = 'temp_workfolder'
    
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    except Exception as e:
        print(f"Error removing folder: {e}")

if __name__ == "__main__":
    remove_temp_folder()



## hier verwijderen we de folder output_plots indien deze nog zou bestaan
import os
import shutil

def remove_temp_folder():
    folder_path = 'output_plots'
    
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    except Exception as e:
        print(f"Error removing folder: {e}")

if __name__ == "__main__":
    remove_temp_folder()



#" hier verwijderen we de oyutput file indien deze nog zou bestaan
import os

# Define the file name
file_name = "oszicar_analysis.png"

# Check if the file exists
if os.path.exists(file_name):
    # Remove the file
    os.remove(file_name)
    print(f"File '{file_name}' has been removed.")
else:
    pass






import os
import shutil

def copy_oszicar_to_temp():
    # Define the temp directory path
    temp_dir = "./temp_workfolder"
    
    # Create temp directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"Created directory: {temp_dir}")
    
    # Copy the OSZICAR file
    try:
        shutil.copy(oszicar, temp_dir)
        print(f"Successfully copied {oszicar} to {temp_dir}")
    except FileNotFoundError:
        print(f"Error: File '{oszicar}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    copy_oszicar_to_temp()




## nu veranderen we de WD
import os

os.chdir('./temp_workfolder')




## nu splitsen we de complete oszicar in de verschillend esteps taken
## de splitsing gebeurt op baiss van lijnen die met "       N       E                     dE             d eps       ncg     rms          rms(c)" starten
## dit is waar voor zowel MD als opt

import os

# Create a directory to store the separated step files
output_dir = "separate_steps"
os.makedirs(output_dir, exist_ok=True)

# Input OSZICAR file
input_file = oszicar

with open(input_file, "r") as file:
    lines = file.readlines()

# Initialize variables
current_step = []
step_count = 0
is_electronic_section = False

for line in lines:
    # Detect the start of a new step based on the header pattern
    if "N       E" in line and "rms(c)" in line:
        # If we have accumulated a step, save it to a file
        if current_step:
            step_count += 1
            output_file = os.path.join(output_dir, f"step_{step_count}.txt")
            with open(output_file, "w") as step_file:
                step_file.writelines(current_step)
            current_step = []

        # Start a new step
        current_step.append(line)
        is_electronic_section = True
    elif is_electronic_section:
        # Add lines to the electronic section until we encounter the ionic relaxation step
        if line.strip() and line[0].isdigit():
            is_electronic_section = False
            current_step.append(line)
        else:
            current_step.append(line)
    else:
        # Add the ionic relaxation step
        current_step.append(line)

# Write the last step to a file, if any
if current_step:
    step_count += 1
    output_file = os.path.join(output_dir, f"step_{step_count}.txt")
    with open(output_file, "w") as step_file:
        step_file.writelines(current_step)

print(f"Separated steps written to the '{output_dir}' directory.")



## nu pakken we alle ionic information en zetten we die in een apparte file
import os
import re

# Define the directory containing the step files
input_dir = "./separate_steps"

output_file = "ionic_steps.txt"

# Regex pattern to identify ionic step lines starting with either F= or T= (het getal ervoor wordt rekening mee gehouden)
ionic_pattern = re.compile(r"^\s*\d+\s+(F=|T=)")

# Get a sorted list of step files
step_files = sorted(

    [f for f in os.listdir(input_dir) if f.startswith("step_") and f.endswith(".txt")],
    key=lambda x: int(x.split("_")[1].split(".")[0]),
)
# Open the output file for writing
with open(output_file, "w") as outfile:

    for step_file in step_files:
        # Construct the full path to the step file
        step_path = os.path.join(input_dir, step_file)
        # Read the file and split lines into matched and unmatched
        with open(step_path, "r") as infile:

            lines = infile.readlines()
        # Separate ionic lines and other lines
        ionic_lines = [line for line in lines if ionic_pattern.match(line)]

        remaining_lines = [line for line in lines if not ionic_pattern.match(line)]
        # Write ionic lines to the output file
        outfile.writelines(ionic_lines)

        # Overwrite the original file with the remaining lines
        with open(step_path, "w") as outfile_step:

            outfile_step.writelines(remaining_lines)
print(f"All ionic step information has been moved to {output_file}. The input files have been updated.")




## nu allignen we alle files naar links en maken we er een soort csv file format van
def process_file(filename):

    try:
        # Read all lines from the file
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Strip whitespace, replace multiple spaces with commas, and add newline
        processed_lines = []
        for line in lines:
            # Strip leading/trailing whitespace
            line = line.strip()
            # Replace multiple spaces with a single space, then replace with comma
            line = ' '.join(line.split())
            line = line.replace(' ', ',')
            processed_lines.append(line + '\n')
        
        # Write the processed lines back to the file
        with open(filename, 'w') as file:
            file.writelines(processed_lines)
            
        print(f"Converted {filename} to CSV format")
        
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
# Execute the function on the specified file
if __name__ == "__main__":

    process_file('ionic_steps.txt')







## de volgende stap is specifiek voor optimizatie oszicars aangezien deze toch iets anders zijn dan MD oszicars
## het volgende script cut kolommen die niet handig of niet juist uitkomen voor een mooie csv file enkel en alleen als er 9 kolommmen in totaal zijn

import csv

# Define the input and output file names
input_file = 'ionic_steps.txt'
output_file = 'ionic_steps.csv'

# Open the input file and read the lines
with open(input_file, 'r') as f:
    lines = f.readlines()

# Check if all lines have exactly 10 columns
if all(len(line.strip().split(',')) == 10 for line in lines):
    print('The file is a geometry optimization.')
    # Open the output file in write mode
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in lines:
            # Split the line by commas
            columns = line.strip().split(',')
            
            # Select only the desired columns (1st, 3rd, 5th, 8th, and 10th)
            selected_columns = [columns[0], columns[2], columns[4], columns[7], columns[9]]
# de enigste kolommen die we willen zijn ""
            
            # Write the selected columns to the CSV file
            writer.writerow(selected_columns)
    print(f"Output saved as {output_file}")
else:
    print('The file is not a geometry optimization.')


## nu doen we hetzelfde als hierboven maar dan voor een MD oszicar waar 17 kolommen zijn
import csv

# Define the input and output file names
input_file = 'ionic_steps.txt'
output_file = 'ionic_steps.csv'

# Open the input file and read the lines
with open(input_file, 'r') as f:
    lines = f.readlines()

# Calculate the number of columns for each line
columns_per_line = [len(line.strip().split(',')) for line in lines]

# Check if all lines have the same number of columns
all_same = all(n == columns_per_line[0] for n in columns_per_line)

if not all_same:
    print('The file is not a MD simulation.')
else:
    num_columns = columns_per_line[0]
    if num_columns not in (15, 17):
        print('The file is not a MD simulation.')
    else:
        print('The file is a MD simulation.')
        # Open the output file in write mode
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for line in lines:
                # Split the line by commas
                columns = line.strip().split(',')
                # Select columns based on the number of columns
                if num_columns == 17:
                    selected_columns = [columns[0], columns[2], columns[4], columns[6], columns[8], columns[10], columns[12], columns[14], columns[16]]
                else:  # 15 columns
                    selected_columns = [columns[0], columns[2], columns[4], columns[6], columns[8], columns[10], columns[12], columns[14]]
                # Write the selected columns to the CSV file
                writer.writerow(selected_columns)
        print(f"Output saved as {output_file}")



# dit is een fix voor de opt maar werkt probleemloos voor md
## nu verwijderen we de '=' symbool
import csv

# Define the file name
file_name = 'ionic_steps.csv'

# Read the file and modify the 4th column
with open(file_name, 'r') as file:

    reader = csv.reader(file)
    rows = []
    for row in reader:
        if len(row) > 3:  # Ensure the row has at least 4 columns
            row[3] = row[3].replace('=', '')  # Remove '=' from the 4th column
        rows.append(row)
# Overwrite the file with the updated content
with open(file_name, 'w', newline='') as file:

    writer = csv.writer(file)
    writer.writerows(rows)
print(f"The '=' symbol has been removed from the 4th column of {file_name}.")





## volgende script is een fix voor een md csv file
## de '.' wordt verwijderd van de 2de kolom wat de temperatuur is
## werkt enkel als er 9 kolommen in toitaal zijn
import csv

file_name = 'ionic_steps.csv'

try:
    # Read and process the file in one pass
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        rows = []
        
        for row in reader:
            # Check if row has 9 columns
            if len(row) == 9 or len(row) == 8:
                # Modify the second column (index 1) by removing periods
                row[1] = row[1].replace('.', '')
                rows.append(row)
            else:
                raise ValueError(f"Found row with {len(row)} columns instead of expected 9")
    
    # Write the modified data back to the same file
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

except FileNotFoundError:
    print(f"Error: Could not find file {file_name}")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
# print("succes")





## nu passen we de getallen in de csv file aan
import re

# File path
file_path = 'ionic_steps.csv'

# Function to add leading zeros to numbers
def add_leading_zeros(line):

    # Regular expression to find numbers starting with a decimal point
    corrected_line = re.sub(r'(?<!\d)\.(\d+)', r'0.\1', line)
    return corrected_line
# Read the file, process each line, and write back the corrected content
with open(file_path, 'r') as file:

    lines = file.readlines()
corrected_lines = [add_leading_zeros(line) for line in lines]

with open(file_path, 'w') as file:
    file.writelines(corrected_lines)

print(f"Corrected numbers in '{file_path}' to floating point numbers.")
#

#
#
##
#
#

## nu voegen we een header toe
import csv

def add_header_based_on_columns(file_name):
    # Define headers for different column counts
    headers = {
        5: "Step_Number,Force_(F),Energy_(E0),Energy_Difference_(dE),Magnetization_(mag)",
        9: "Step_Number,Temeprature_[K],Energy_[eV],Force[eV/A],E0_[eV],Ek_[eV],SP,SK,magnetisation",
        8: "Step_Number,Temeprature_[K],Energy_[eV],Force[eV/A],E0_[eV],Ek_[eV],SP,SK"
    }
    
    try:
        # Read the first line to determine the number of columns
        with open(file_name, "r") as file:
            first_line = file.readline().strip()
            # Count the number of columns in the first line
            num_columns = len(first_line.split(','))
            # Read the rest of the content
            remaining_content = file.readlines()
        
        # Check if the number of columns matches our defined headers
        if num_columns not in headers:
            raise ValueError(f"File has {num_columns} columns. Only files with 5 (spin polarized geometry optimization), 8 (non-spin polarized MD run) or 9 (spin polarized MD run) columns are supported.")
        
        # Write the appropriate header and content back to the file
        with open(file_name, "w") as file:
            file.write(headers[num_columns] + "\n")
            file.write(first_line + "\n")  # Write back the first line
            file.writelines(remaining_content)  # Write the rest of the content
        
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# Usage
file_name = "ionic_steps.csv"
add_header_based_on_columns(file_name)






## alright nu passen we de electronic steps aan
#" hier maken we van alle files csv format

import os
import glob
def process_file(input_file, output_file):

    with open(input_file, 'r') as f:
        # Skip the header line
        lines = f.readlines()[1:]
    
    # Process each line
    processed_lines = []
    for line in lines:
        # Split by whitespace and join with commas
        values = [val for val in line.split() if val]  # Remove empty strings
        processed_line = ','.join(values)
        processed_lines.append(processed_line)
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(processed_lines))
def main():
    # Create output directory if it doesn't exist

    output_dir = './electronic_steps_csv'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process all step files
    input_pattern = './separate_steps/step_*.txt'
    for input_file in glob.glob(input_pattern):
        # Generate output filename
        basename = os.path.basename(input_file)
        name_without_ext = os.path.splitext(basename)[0]
        output_file = os.path.join(output_dir, f"{name_without_ext}.csv")
        
        # Process the file
        process_file(input_file, output_file)
if __name__ == "__main__":
    main()




## nu passen we de csv files aan
import os

import pandas as pd
from pathlib import Path
def clean_csv_files(directory):
    # Create Path object for the directory

    dir_path = Path(directory)
    
    # Check if directory exists
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist!")
        return
    
    # Get all CSV files in the directory
    csv_files = list(dir_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return
    
    for file_path in csv_files:
        try:
            # Read the file content
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Process each line to remove ':' from the first column
            cleaned_lines = []
            for line in lines:
                # Split the line into columns
                columns = line.split(',')
                # Remove ':' from the first column
                columns[0] = columns[0].replace(':', '')
                # Join the columns back together
                cleaned_line = ','.join(columns)
                cleaned_lines.append(cleaned_line)
            
            # Write the cleaned content back to the file
            with open(file_path, 'w') as file:
                file.writelines(cleaned_lines)
            
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
# Execute the script
if __name__ == "__main__":

    directory = "./electronic_steps_csv"
    clean_csv_files(directory)



## hier voegen we headers toe aan de files
import os

from tempfile import NamedTemporaryFile
import shutil
def add_header_to_files(directory):

    # Header to add
    header = "Algorithm,N,E,dE,d_eps,ncg,rms,rms(c)\n"
    
    # Ensure directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist")
    
    # Process each file in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip if it's not a file
        if not os.path.isfile(filepath):
            continue
            
        # Create a temporary file
        temp_file = NamedTemporaryFile(mode='w', delete=False)
        
        try:
            # Write the header first
            temp_file.write(header)
            
            # Copy the original content
            with open(filepath, 'r') as original_file:
                temp_file.write(original_file.read())
                
            temp_file.close()
            
            # Replace the original file with the new one
            shutil.move(temp_file.name, filepath)
            
        except Exception as e:
            # Clean up the temporary file if something goes wrong
            os.unlink(temp_file.name)
            raise e
if __name__ == "__main__":
    try:

        add_header_to_files("electronic_steps_csv")
        print("Successfully added header to all files in electronic_steps_csv")
    except Exception as e:
        print(f"An error occurred: {e}")




# nu verwijderen we seperate_steps omdat dit wel nogal zwaar kan zijn
shutil.rmtree('separate_steps')





###########################################################################
## Dit is de gist van het script waar alle informatie wordt geprocessed
## hiervoor hebben we enkel ons bezig gehouden met het creeeren van goede files om mee te werken
## de volgende stappen worden gesplitst op basis of er 5 kolommen (geo opt) of 9 zijn (md simulatie) 



import pandas as pd
import numpy as np


df = pd.read_csv('ionic_steps.csv')

# Check if the number of columns is exactly 5########## GEO OPT
if len(df.columns) == 5:
    print("Analysis accoring to a spin polarized geometry optimization will be shown ...")
    

    import pandas as pd

    import os
    import matplotlib.pyplot as plt
    from pathlib import Path
    def analyze_vasp_optimization():
        # Read ionic steps data
        ionic_data = pd.read_csv('./ionic_steps.csv')

        # Initialize storage for electronic step analysis
        electronic_analysis = []
        total_electronic_steps = 0  # Counter for total electronic steps

        # Analyze each electronic step file
        for step in range(1, len(ionic_data) + 1):
            file_path = f'./electronic_steps_csv/step_{step}.csv'
            if os.path.exists(file_path):
                e_step_data = pd.read_csv(file_path)
                num_steps = len(e_step_data)
                total_electronic_steps += num_steps  # Add to total counter

                analysis = {
                    'ionic_step': step,
                    'num_electronic_steps': num_steps,
                    'initial_energy': e_step_data['E'].iloc[0],
                    'final_energy': e_step_data['E'].iloc[-1],
                    'energy_convergence': abs(e_step_data['dE'].iloc[-1]),
                    'max_rms': e_step_data['rms'].max(),
                    'final_rms': e_step_data['rms'].iloc[-1]
                }
                electronic_analysis.append(analysis)

        e_analysis_df = pd.DataFrame(electronic_analysis)

        # Generate analysis report
        report = {
            'total_ionic_steps': len(ionic_data),
            'total_electronic_steps': total_electronic_steps,  # Add to report
            'final_energy': ionic_data['Energy_(E0)'].iloc[-1],
            'energy_convergence': abs(ionic_data['Energy_Difference_(dE)'].iloc[-1]),
            'force_convergence': abs(ionic_data['Force_(F)'].iloc[-1]),
            'max_force': ionic_data['Force_(F)'].abs().max(),
            'avg_electronic_steps': e_analysis_df['num_electronic_steps'].mean(),
            'max_electronic_steps': e_analysis_df['num_electronic_steps'].max(),
            'final_magnetization': ionic_data['Magnetization_(mag)'].iloc[-1]
        }

        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Energy convergence plot
        ax1.plot(ionic_data['Step_Number'], ionic_data['Energy_(E0)'], 'b-o')
        ax1.set_xlabel('Ionic Step')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title('Energy Convergence')

        # Force convergence plot
        ax2.plot(ionic_data['Step_Number'], abs(ionic_data['Force_(F)']), 'r-o')
        ax2.set_xlabel('Ionic Step')
        ax2.set_ylabel('|Force| (eV/Å)')
        ax2.set_title('Force Convergence')

        # Electronic steps per ionic step
        ax3.bar(e_analysis_df['ionic_step'], e_analysis_df['num_electronic_steps'])
        ax3.set_xlabel('Ionic Step')
        ax3.set_ylabel('Number of Electronic Steps')
        ax3.set_title('Electronic Steps per Ionic Step')

        plt.tight_layout()

        return report, ionic_data, e_analysis_df, fig
    def print_report(report):
        """Prints a formatted analysis report"""

        print("\n=== VASP Structural Optimization Analysis ===")
        print(f"\nTotal Ionic Steps: {report['total_ionic_steps']}")
        print(f"Final Energy: {report['final_energy']:.6f} eV")
        print(f"Final Energy Convergence: {report['energy_convergence']:.6e} eV")

        print(f"Final Force Convergence: {report['force_convergence']:.6e} eV/Å")
        print(f"Maximum Force: {report['max_force']:.6e} eV/Å")
        print(f"\nElectronic Step Statistics:")
        print(f"Total Electronic Steps: {report['total_electronic_steps']}")  # Added to print output
        print(f"Average Electronic Steps per Ionic Step: {report['avg_electronic_steps']:.1f}")
        print(f"Maximum Electronic Steps in any Ionic Step: {report['max_electronic_steps']}")
        print(f"\nFinal Magnetization: {report['final_magnetization']:.4f}")
    if __name__ == "__main__":
        # Run analysis

        report, ionic_data, e_analysis_df, fig = analyze_vasp_optimization()

        # Print report
        print_report(report)

        # Save visualization
        fig.savefig('oszicar_analysis.png', dpi=resolution_plots)

        # Save detailed data to CSV files
        ionic_data.to_csv('ionic_analysis.csv', index=False)
        e_analysis_df.to_csv('electronic_analysis.csv', index=False)


    ## hier moven we de file naar de parent
    destination_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
    destination_file = os.path.join(destination_directory, 'oszicar_analysis.png')
    # Move the file
    shutil.move('oszicar_analysis.png', destination_file)




    ## nu doen we een comparisson van verschillende SCF methoden in de electronic steps
    import os
    import csv
    import glob
    import matplotlib.pyplot as plt

    # Folder where CSV files are stored
    folder = 'electronic_steps_csv'
    pattern = os.path.join(folder, 'step_*.csv')
    files = glob.glob(pattern)

    # Dictionaries to store counts and cumulative ncg values
    counts = {}
    ncg_totals = {}

    for filepath in files:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Skip header rows (which typically contain 'Algorithm')
                if row[0].strip().lower() == 'algorithm':
                    continue

                # Extract algorithm type (first column) and ncg value (6th column)
                algo = row[0].strip()
                try:
                    # The ncg value is the 6th column (index 5)
                    ncg_value = float(row[5])
                except (IndexError, ValueError):
                    # Skip rows that don't have expected format
                    continue

                # Update counts and ncg totals
                counts[algo] = counts.get(algo, 0) + 1
                ncg_totals[algo] = ncg_totals.get(algo, 0) + ncg_value

    # Compute average ncg for each algorithm
    avg_ncg = {algo: ncg_totals[algo] / counts[algo] for algo in counts}

    # Create a figure with two subplots: a pie chart and a bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Pie Chart: Distribution of SCF method counts
    labels = list(counts.keys())
    sizes = [counts[label] for label in labels]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('SCF Method Count Distribution')

    # Bar Chart: Average ncg values (as a proxy for time per step)
    algos = list(avg_ncg.keys())
    avg_values = [avg_ncg[algo] for algo in algos]
    ax2.bar(algos, avg_values, color='skyblue')
    ax2.set_title('Average "ncg" per SCF Method')
    ax2.set_ylabel('Average ncg')
    ax2.set_xlabel('SCF Method')

    plt.suptitle('Comparison of SCF Methods in Electronic Convergence')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('scf_algorithm_comparison.png', dpi=resolution_plots)




    # nu moven we deze naar de output_plots
    import os
    import shutil

    destination_path = os.path.join('output_plots', os.path.basename('scf_algorithm_comparison.png'))
    shutil.move('scf_algorithm_comparison.png', 'output_plots')









# Check if the number of columns is exactly 9########## MD
if len(df.columns) == 9:
    print("Analysis accoring to a spin polarized molecular dynamics will be shown ...")

    #!/usr/bin/env python3
    """
    Combined MD Simulation Analysis Script
    Analyzes ionic steps and electronic SCF convergence data from VASP simulations
    Combines functionality from both the main analyzer and SCF steps analysis
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import glob
    import re
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')

    class MDAnalyzer:
        def __init__(self, ionic_file='ionic_steps.csv', electronic_folder='electronic_steps_csv', 
                     output_folder='output_plots', dpi=resolution_plots):
            """
            Initialize the MD Analyzer

            Args:
                ionic_file (str): Path to the ionic steps CSV file
                electronic_folder (str): Path to folder containing electronic step CSV files
                output_folder (str): Folder to save output graphs
                dpi (int): DPI for saved figures
            """
            self.ionic_file = ionic_file
            self.electronic_folder = electronic_folder
            self.output_folder = output_folder
            self.dpi = dpi
            self.ionic_data = None
            self.electronic_data = {}

            # Create output folder if it doesn't exist
            os.makedirs(self.output_folder, exist_ok=True)

        def extract_step_number(self, filename):
            """Extract step number from filename like 'step_1.csv'"""
            match = re.search(r'step_(\d+)\.csv', filename)
            if match:
                return int(match.group(1))
            return None

        def load_ionic_data(self):
            """Load and clean ionic steps data"""
            try:
                self.ionic_data = pd.read_csv(self.ionic_file)

                # Clean column names (remove spaces and special characters)
                self.ionic_data.columns = self.ionic_data.columns.str.strip()

                # Select only important columns
                important_cols = ['Step_Number', 'Temeprature_[K]', 'Energy_[eV]', 
                                'Force[eV/A]', 'E0_[eV]', 'Ek_[eV]']

                # Check which columns actually exist
                existing_cols = [col for col in important_cols if col in self.ionic_data.columns]
                self.ionic_data = self.ionic_data[existing_cols]

                print(f"Loaded ionic data with {len(self.ionic_data)} steps")
                print(f"Available columns: {list(self.ionic_data.columns)}")

            except FileNotFoundError:
                print(f"Error: Could not find {self.ionic_file}")
            except Exception as e:
                print(f"Error loading ionic data: {e}")

        def load_electronic_data(self):
            """Load electronic SCF convergence data from all step files"""
            if not os.path.exists(self.electronic_folder):
                print(f"Error: Electronic folder {self.electronic_folder} not found")
                return

            # Find all step_*.csv files
            pattern = os.path.join(self.electronic_folder, "step_*.csv")
            step_files = glob.glob(pattern)

            if not step_files:
                print(f"No step_*.csv files found in {self.electronic_folder}")
                return

            print(f"Found {len(step_files)} electronic step files")

            for file_path in step_files:
                try:
                    # Extract step number from filename
                    filename = os.path.basename(file_path)
                    step_num = self.extract_step_number(filename)

                    if step_num is None:
                        continue

                    # Load electronic data
                    df = pd.read_csv(file_path)

                    # Select important columns
                    important_cols = ['Algorithm', 'N', 'E', 'dE', 'd_eps', 'ncg']
                    existing_cols = [col for col in important_cols if col in df.columns]

                    self.electronic_data[step_num] = df[existing_cols]

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

            print(f"Successfully loaded electronic data for {len(self.electronic_data)} steps")

        def analyze_ionic_convergence(self):
            """Analyze ionic step convergence"""
            if self.ionic_data is None:
                print("No ionic data loaded")
                return

            print("\n=== IONIC CONVERGENCE ANALYSIS ===")
            print(f"Total ionic steps: {len(self.ionic_data)}")

            # Basic statistics
            if 'Energy_[eV]' in self.ionic_data.columns:
                final_energy = self.ionic_data['Energy_[eV]'].iloc[-1]
                energy_change = abs(self.ionic_data['Energy_[eV]'].iloc[-1] - self.ionic_data['Energy_[eV]'].iloc[0])
                print(f"Final energy: {final_energy:.6f} eV")
                print(f"Total energy change: {energy_change:.6f} eV")

            if 'Force[eV/A]' in self.ionic_data.columns:
                final_force = self.ionic_data['Force[eV/A]'].iloc[-1]
                print(f"Final force: {final_force:.6f} eV/Å")

            if 'Temeprature_[K]' in self.ionic_data.columns:
                avg_temp = self.ionic_data['Temeprature_[K]'].mean()
                temp_std = self.ionic_data['Temeprature_[K]'].std()
                print(f"Average temperature: {avg_temp:.1f} ± {temp_std:.1f} K")

        def analyze_electronic_convergence(self):
            """Analyze electronic SCF convergence for each step"""
            if not self.electronic_data:
                print("No electronic data loaded")
                return {}

            print("\n=== ELECTRONIC CONVERGENCE ANALYSIS ===")

            convergence_stats = {}

            for step_num, df in self.electronic_data.items():
                if len(df) == 0:
                    continue

                # Number of SCF iterations
                scf_iterations = len(df)

                # Final energy and energy convergence
                if 'E' in df.columns:
                    final_energy = df['E'].iloc[-1]
                    if 'dE' in df.columns and len(df) > 1:
                        final_de = abs(df['dE'].iloc[-1])
                    else:
                        final_de = None
                else:
                    final_energy = None
                    final_de = None

                convergence_stats[step_num] = {
                    'scf_iterations': scf_iterations,
                    'final_energy': final_energy,
                    'final_de': final_de
                }

            # Summary statistics
            iterations = [stats['scf_iterations'] for stats in convergence_stats.values()]
            avg_iterations = np.mean(iterations)
            max_iterations = np.max(iterations)
            min_iterations = np.min(iterations)

            print(f"Average SCF iterations per step: {avg_iterations:.1f}")
            print(f"Max SCF iterations: {max_iterations}")
            print(f"Min SCF iterations: {min_iterations}")
            print(f"SCF steps range: {min_iterations} - {max_iterations}")

            # Steps with convergence issues (high iteration count)
            problematic_steps = [step for step, stats in convergence_stats.items() 
                               if stats['scf_iterations'] > avg_iterations + 2*np.std(iterations)]

            if problematic_steps:
                print(f"Steps with potential convergence issues: {problematic_steps}")

            return convergence_stats

        def plot_ionic_evolution(self):
            """Plot ionic step evolution"""
            if self.ionic_data is None:
                print("No ionic data to plot")
                return

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Ionic Step Evolution', fontsize=16)

            step_numbers = self.ionic_data['Step_Number']

            # Energy evolution
            if 'Energy_[eV]' in self.ionic_data.columns:
                axes[0,0].plot(step_numbers, self.ionic_data['Energy_[eV]'], 'b-o', markersize=3)
                axes[0,0].set_xlabel('Step Number')
                axes[0,0].set_ylabel('Energy (eV)')
                axes[0,0].set_title('Total Energy')
                axes[0,0].grid(True, alpha=0.3)

            # Force evolution
            if 'Force[eV/A]' in self.ionic_data.columns:
                axes[0,1].plot(step_numbers, self.ionic_data['Force[eV/A]'], 'r-o', markersize=3)
                axes[0,1].set_xlabel('Step Number')
                axes[0,1].set_ylabel('Force (eV/Å)')
                axes[0,1].set_title('Forces')
                axes[0,1].grid(True, alpha=0.3)

            # Temperature evolution
            if 'Temeprature_[K]' in self.ionic_data.columns:
                axes[1,0].plot(step_numbers, self.ionic_data['Temeprature_[K]'], 'g-o', markersize=3)
                axes[1,0].set_xlabel('Step Number')
                axes[1,0].set_ylabel('Temperature (K)')
                axes[1,0].set_title('Temperature')
                axes[1,0].grid(True, alpha=0.3)

            # Kinetic energy evolution
            if 'Ek_[eV]' in self.ionic_data.columns:
                axes[1,1].plot(step_numbers, self.ionic_data['Ek_[eV]'], 'm-o', markersize=3)
                axes[1,1].set_xlabel('Step Number')
                axes[1,1].set_ylabel('Kinetic Energy (eV)')
                axes[1,1].set_title('Kinetic Energy')
                axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save the plot
            plt.savefig(os.path.join(self.output_folder, 'ionic_evolution.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.show()

        def plot_scf_convergence(self, step_numbers=None, max_plots=6):
            """Plot SCF convergence for selected steps"""
            if not self.electronic_data:
                print("No electronic data to plot")
                return

            if step_numbers is None:
                # Select first, last, and some intermediate steps
                all_steps = sorted(self.electronic_data.keys())
                if len(all_steps) <= max_plots:
                    step_numbers = all_steps
                else:
                    # Select evenly spaced steps
                    indices = np.linspace(0, len(all_steps)-1, max_plots, dtype=int)
                    step_numbers = [all_steps[i] for i in indices]

            n_plots = len(step_numbers)
            cols = min(3, n_plots)
            rows = (n_plots + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            if n_plots == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle('SCF Convergence for Selected Steps', fontsize=16)

            for i, step_num in enumerate(step_numbers):
                row = i // cols
                col = i % cols
                ax = axes[row, col] if rows > 1 else axes[col]

                if step_num not in self.electronic_data:
                    ax.text(0.5, 0.5, f'No data for step {step_num}', 
                           ha='center', va='center', transform=ax.transAxes)
                    continue

                df = self.electronic_data[step_num]

                if 'E' in df.columns:
                    ax.plot(df['N'], df['E'], 'b-o', markersize=4)
                    ax.set_xlabel('SCF Iteration')
                    ax.set_ylabel('Energy (eV)')
                    ax.set_title(f'Step {step_num} - SCF Convergence')
                    ax.grid(True, alpha=0.3)

                    # Add final energy as text
                    final_energy = df['E'].iloc[-1]
                    ax.text(0.05, 0.95, f'Final E: {final_energy:.6f} eV', 
                           transform=ax.transAxes, fontsize=10, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Hide empty subplots
            for i in range(n_plots, rows * cols):
                row = i // cols
                col = i % cols
                if rows > 1:
                    axes[row, col].set_visible(False)
                else:
                    axes[col].set_visible(False)

            plt.tight_layout()

            # Save the plot
            plt.savefig(os.path.join(self.output_folder, 'scf_convergence.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.show()



        def plot_scf_steps_per_md_step(self):
            """Create dedicated plot for SCF steps per MD step (from second script)"""
            if not self.electronic_data:
                print("No electronic data to plot")
                return

            # Collect data
            step_data = []
            for step_num, df in self.electronic_data.items():
                if len(df) > 0:
                    step_data.append((step_num, len(df)))

            if not step_data:
                print("No valid step data found!")
                return

            # Sort by step number
            step_data.sort(key=lambda x: x[0])

            # Extract step numbers and SCF counts
            step_numbers = [data[0] for data in step_data]
            scf_counts = [data[1] for data in step_data]

            # Create the plot
            plt.figure(figsize=(12, 8))

            # Main plot
            plt.plot(step_numbers, scf_counts, 'b-', linewidth=1.5, alpha=0.7, label='SCF steps')

            # Add horizontal line for average
            avg_scf = np.mean(scf_counts)
            plt.axhline(y=avg_scf, color='green', linestyle='--', alpha=0.8, 
                        label=f'Average: {avg_scf:.1f}')

            # Customize the plot
            plt.xlabel('MD Step Number', fontsize=12)
            plt.ylabel('Number of SCF Steps', fontsize=12)
            plt.title('SCF Convergence Steps per MD Trajectory Step', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Set integer ticks for y-axis if reasonable
            if max(scf_counts) - min(scf_counts) < 50:
                plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # Adjust layout
            plt.tight_layout()

            # Save the plot
            output_file = os.path.join(self.output_folder, "scf_steps_per_md_step.png")
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to: {output_file}")
            plt.show()

        def generate_report(self):
            """Generate a comprehensive analysis report"""
            print("\n" + "="*60)
            print("           MD SIMULATION ANALYSIS REPORT")
            print("="*60)

            # Load data if not already loaded
            if self.ionic_data is None:
                self.load_ionic_data()
            if not self.electronic_data:
                self.load_electronic_data()

            # Analyze convergence
            self.analyze_ionic_convergence()
            convergence_stats = self.analyze_electronic_convergence()

            # Generate plots
            print("\nGenerating plots...")

            # Plot ionic evolution if ionic data is available
            if self.ionic_data is not None:
                self.plot_ionic_evolution()

            # Plot electronic convergence data if available
            if self.electronic_data:
                self.plot_scf_convergence()
                self.plot_scf_steps_per_md_step()

            print(f"\nAll plots saved in: {self.output_folder}")
            print("Analysis complete!")

    def main():
        """Main function to run the analysis"""
        # Initialize analyzer with custom parameters
        analyzer = MDAnalyzer(dpi=resolution_plots)  # You can adjust DPI here

        # Generate full report
        analyzer.generate_report()

        # Example of how to use individual methods:
        # analyzer = MDAnalyzer(output_folder='my_graphs', dpi=150)
        # analyzer.load_ionic_data()
        # analyzer.load_electronic_data()
        # analyzer.plot_ionic_evolution()
        # analyzer.plot_scf_convergence(step_numbers=[1, 5, 10])
        # analyzer.plot_scf_steps_per_md_step()

    if __name__ == "__main__":
        main()









    ## nu moven we de output folder naar de parent
    import os
    import shutil
    from pathlib import Path

    def move_to_parent_directory(folder_name="output_plots"):
        try:
            # Get the current working directory
            current_dir = Path.cwd()

            # Define source and destination paths
            source_path = current_dir / folder_name
            destination_path = current_dir.parent / folder_name

            # Check if source folder exists
            if not source_path.exists():
                print(f"Error: Folder '{folder_name}' not found in current directory")
                return False

            # Check if a folder with same name already exists in parent directory
            if destination_path.exists():
                print(f"Error: Folder '{folder_name}' already exists in parent directory")
                return False

            # Move the folder to parent directory
            shutil.move(str(source_path), str(destination_path))
            return True

        except PermissionError:
            print("Error: Permission denied. Check folder permissions.")
            return False
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False

    if __name__ == "__main__":
        move_to_parent_directory()




            


# Check if the number of columns is exactly 8########## MD (non-spin polarized MD run)
if len(df.columns) == 8:
    print("Analysis accoring to a non-spin polarized molecular dynamics will be shown ...")

    ## ELECTRONIC ANALYSIS FOR MD
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Define the folder containing the CSV files
    folder_path = "./electronic_steps_csv"
    output_folder = "./output_plots"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    def analyze_electronic_steps(folder_path):
        # Initialize variables to store analysis results
        total_steps = 0
        max_steps = 0
        steps_per_md = []

        # Prepare lists for visualization
        all_energy = []
        all_dE = []
        all_d_eps = []
        all_rms = []
        all_rms_c = []

        # Process each CSV file in the folder
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".csv") and file.startswith("step_"):
                # Read the file
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)

                # Number of electronic steps in this MD step
                num_steps = len(df)
                steps_per_md.append(num_steps)

                # Update totals and max
                total_steps += num_steps
                max_steps = max(max_steps, num_steps)

                # Append data for visualization
                all_energy.extend(df["E"])
                all_dE.extend(df["dE"])
                all_d_eps.extend(df["d_eps"])
                all_rms.extend(df["rms"])
                all_rms_c.extend(df["rms(c)"])

        # Calculate average number of steps
        avg_steps = total_steps / len(steps_per_md)

        # Print summary
        print(f"Total MD steps analyzed: {len(steps_per_md)}")
        print(f"Total electronic relaxation steps: {total_steps}")
        print(f"Maximum steps in a single MD step: {max_steps}")
        print(f"Average steps per MD step: {avg_steps:.2f}")

        # Plot and save histogram of steps per MD step
        plt.figure(figsize=(10, 6))
        plt.hist(steps_per_md, bins=range(1, max_steps + 2), edgecolor='black', alpha=0.7)
        plt.title("Histogram of Electronic Relaxation Steps per MD Step")
        plt.xlabel("Number of Electronic Relaxation Steps")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_folder, "histogram_steps_per_md.png"), dpi=resolution_plots)
        plt.close()

        # Plot and save trends for key quantities
        plt.figure(figsize=(10, 6))
        plt.plot(all_energy, label='Energy (E)', alpha=0.7)
        plt.title("Trend of Energy (E) During Electronic Relaxation")
        plt.xlabel("Relaxation Step")
        plt.ylabel("Energy (E)")
        plt.legend()
        plt.savefig(os.path.join(output_folder, "trend_energy.png"), dpi=resolution_plots)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(all_dE, label='Energy Change (dE)', alpha=0.7)
        plt.title("Trend of Energy Change (dE) During Electronic Relaxation")
        plt.xlabel("Relaxation Step")
        plt.ylabel("Energy Change (dE)")
        plt.legend()
        plt.savefig(os.path.join(output_folder, "trend_dE.png"), dpi=resolution_plots)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(all_d_eps, label='Change in Dielectric Constant (d_eps)', alpha=0.7)
        plt.title("Trend of Dielectric Change (d_eps) During Electronic Relaxation")
        plt.xlabel("Relaxation Step")
        plt.ylabel("d_eps")
        plt.legend()
        plt.savefig(os.path.join(output_folder, "trend_d_eps.png"), dpi=resolution_plots)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(all_rms, label='RMS', alpha=0.7)
        plt.plot(all_rms_c, label='RMS(c)', alpha=0.7)
        plt.title("RMS and RMS(c) During Electronic Relaxation")
        plt.xlabel("Relaxation Step")
        plt.ylabel("RMS / RMS(c)")
        plt.legend()
        plt.savefig(os.path.join(output_folder, "trend_rms.png"), dpi=resolution_plots)
        plt.close()

    # Run the analysis
    analyze_electronic_steps(folder_path)



    ## nu doen we een comparisson van verschillende SCF methoden in de electronic steps
    import os
    import csv
    import glob
    import matplotlib.pyplot as plt

    # Folder where CSV files are stored
    folder = 'electronic_steps_csv'
    pattern = os.path.join(folder, 'step_*.csv')
    files = glob.glob(pattern)

    # Dictionaries to store counts and cumulative ncg values
    counts = {}
    ncg_totals = {}

    for filepath in files:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Skip header rows (which typically contain 'Algorithm')
                if row[0].strip().lower() == 'algorithm':
                    continue

                # Extract algorithm type (first column) and ncg value (6th column)
                algo = row[0].strip()
                try:
                    # The ncg value is the 6th column (index 5)
                    ncg_value = float(row[5])
                except (IndexError, ValueError):
                    # Skip rows that don't have expected format
                    continue

                # Update counts and ncg totals
                counts[algo] = counts.get(algo, 0) + 1
                ncg_totals[algo] = ncg_totals.get(algo, 0) + ncg_value

    # Compute average ncg for each algorithm
    avg_ncg = {algo: ncg_totals[algo] / counts[algo] for algo in counts}

    # Create a figure with two subplots: a pie chart and a bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Pie Chart: Distribution of SCF method counts
    labels = list(counts.keys())
    sizes = [counts[label] for label in labels]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('SCF Method Count Distribution')

    # Bar Chart: Average ncg values (as a proxy for time per step)
    algos = list(avg_ncg.keys())
    avg_values = [avg_ncg[algo] for algo in algos]
    ax2.bar(algos, avg_values, color='skyblue')
    ax2.set_title('Average "ncg" per SCF Method')
    ax2.set_ylabel('Average ncg')
    ax2.set_xlabel('SCF Method')

    plt.suptitle('Comparison of SCF Methods in Electronic Convergence')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('scf_algorithm_comparison.png', dpi=resolution_plots)
    
    
    
    # nu moven we deze naar de output_plots
    import os
    import shutil
    
    destination_path = os.path.join('output_plots', os.path.basename('scf_algorithm_comparison.png'))
    shutil.move('scf_algorithm_comparison.png', 'output_plots')




    ## IONIC ANALYSIS FOR MD
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # Ensure the output directory exists
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    data = pd.read_csv("ionic_steps.csv")

    # Print basic information about the data
    print("Summary of the data:")
    print(data.describe())

    # Generate and save figures
    # 1. Energy vs. Step Number
    plt.figure()
    plt.plot(data["Step_Number"], data["Energy_[eV]"], marker='o', label='Total Energy')
    plt.xlabel("Step Number")
    plt.ylabel("Energy (eV)")
    plt.title("Energy vs. Step Number")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "energy_vs_step.png"), dpi=resolution_plots)
    plt.close()

    # 3. Kinetic Energy vs. Step Number
    plt.figure()
    plt.plot(data["Step_Number"], data["Ek_[eV]"], marker='^', color='green', label='Kinetic Energy')
    plt.xlabel("Step Number")
    plt.ylabel("Kinetic Energy (eV)")
    plt.title("Kinetic Energy vs. Step Number")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "kinetic_energy_vs_step.png"), dpi=resolution_plots)
    plt.close()

    # 4. SP and SK vs. Step Number
    plt.figure()
    plt.plot(data["Step_Number"], data["SP"], label='SP', marker='o', linestyle='--')
    plt.plot(data["Step_Number"], data["SK"], label='SK', marker='x', linestyle=':')
    plt.xlabel("Step Number")
    plt.ylabel("SP / SK")
    plt.title("SP and SK vs. Step Number")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "sp_sk_vs_step.png"))
    plt.close()

    # 5. Forces vs. Step Number
    plt.figure()
    plt.plot(data["Step_Number"], data["Force[eV/A]"], marker='D', color='red', label='Force')
    plt.xlabel("Step Number")
    plt.ylabel("Force (eV/A)")
    plt.title("Force vs. Step Number")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "force_vs_step.png"), dpi=resolution_plots)
    plt.close()

    print("Plots saved in the 'output_plots' directory.")



    ## nu plotten we de temperatuur ifv de step number
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    import os
    
    # Define the file paths
    input_file = "ionic_steps.csv"
    output_folder = "output_plots"
    output_file_png = os.path.join(output_folder, "temperature_vs_step.png")
    output_file_html = os.path.join(output_folder, "temperature_vs_step.html")
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the CSV file
    data = pd.read_csv(input_file)
    
    # Extract the Step Number and Temperature columns
    step_number = data.iloc[:, 0]  # First column
    temperature = data.iloc[:, 1]  # Second column
    
    # Plot the data as PNG
    plt.figure(figsize=(10, 6))
    plt.plot(step_number, temperature, marker="o", linestyle="-", color="red")
    plt.title("Temperature vs Step Number")
    plt.xlabel("Step Number")
    plt.ylabel("Temperature [K]")
    plt.grid(True)
    
    # Save the PNG plot
    plt.savefig(output_file_png)
    plt.close()
    
    # Create an interactive plot using Plotly
    fig = px.line(
        x=step_number,
        y=temperature,
        labels={"x": "Step Number", "y": "Temperature [K]"},
        title="Temperature vs Step Number",
        line_shape="linear",
    )
    fig.update_traces(line_color="red")
    
    # Save the interactive plot as an HTML file
    fig.write_html(output_file_html)
    
    print(f"Graph saved successfully in: {output_file_png}")
    print(f"Interactive graph saved successfully in: {output_file_html}")
    








    ## nu moven we de output folder naar de parent
    import os
    import shutil
    from pathlib import Path

    def move_to_parent_directory(folder_name="output_plots"):
        try:
            # Get the current working directory
            current_dir = Path.cwd()

            # Define source and destination paths
            source_path = current_dir / folder_name
            destination_path = current_dir.parent / folder_name

            # Check if source folder exists
            if not source_path.exists():
                print(f"Error: Folder '{folder_name}' not found in current directory")
                return False

            # Check if a folder with same name already exists in parent directory
            if destination_path.exists():
                print(f"Error: Folder '{folder_name}' already exists in parent directory")
                return False

            # Move the folder to parent directory
            shutil.move(str(source_path), str(destination_path))
            return True

        except PermissionError:
            print("Error: Permission denied. Check folder permissions.")
            return False
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False

    if __name__ == "__main__":
        move_to_parent_directory()













## hier veranderen we de WD naar de parent
# Get the current working directory
current_directory = os.getcwd()

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Change the working directory to the parent directory
os.chdir(parent_directory)


import shutil
import os

# Define the folder name
folder_name = 'temp_workfolder'

# Check if the folder exists
if os.path.exists(folder_name) and os.path.isdir(folder_name):

    # Remove the folder and all its contents
    shutil.rmtree(folder_name)





