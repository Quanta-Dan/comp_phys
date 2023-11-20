import functions
from functions import *
import pandas as pd

file_path = 'parameters.csv'



def check_columns_defined(df):
    required_columns = ['D', 'N','L_over_r', 'mu']

    for col in required_columns:
        assert col in df.columns, f"Column '{col}' is not defined in the CSV file."


# Read the CSV file into a DataFrame
try:
    input_param = pd.read_csv(file_path, header=0)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: File '{file_path}' is empty.")
    exit(1)
except pd.errors.ParserError:
    print(f"Error: Unable to parse file '{file_path}'. Check if it's a valid CSV file.")
    exit(1)



#check all columns are defined
check_columns_defined(input_param)
#import values from csv
L_over_r = input_param["L_over_r"].iloc[0]
N = input_param["N"].iloc[0]
epsilon = L_over_r/N
D = input_param["D"].iloc[0]
mu = input_param["mu"].iloc[0]
epsilon_2 = epsilon**2
#broadcast values to functions
functions.N = N
functions.D = D
functions.mu = mu
functions.epsilon_2 = epsilon_2

#calculate lowest eigenvalue and eigenvector
lowest_EigValue, lowest_EigVector = lowest_eigenvalue_vector(hamiltonian_function,  power_method_tolerance = 1.e-12, conjugate_gradient_tolerance = 1.e-12, max_iters_power_method = 10000, max_iters_conjugate_gradient= 10000)

# #write eigenvector to csv
df_EigVector = pd.DataFrame(lowest_EigVector)
df_EigVector.to_csv(f'eigenvector_{D, N, L_over_r, mu}.csv', header=None, index=None, sep = ',')