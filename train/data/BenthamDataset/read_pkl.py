import pickle

# Path to the pkl file
##pkl_file_path = '../read2016/lines/labels.pkl'

# Load the pkl file
#with open(pkl_file_path, 'rb') as f:
 #   data = pickle.load(f)

# Now `data` contains the dictionary with 'ground_truth' and 'charset'
##print(data)  # Print the entire content

# If you want to access specific parts, you can do so like this:
##ground_truth = data.get('ground_truth', {})
##charset = data.get('charset', [])

##print("Ground Truth:", ground_truth)
##print("Charset:", charset)


#another code 

import pickle

with open("lines/labels.pkl", "rb") as f:
    data = pickle.load(f)
    print(data)
