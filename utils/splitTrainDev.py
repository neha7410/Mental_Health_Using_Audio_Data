import csv
import random
import shutil

COLUMN_HEADER = "participant_id"
PARTITIONS = ["dev", "train", "test"]
PARTITION_EXTENSION = "_split.csv"
ID_EXTENSION = "_merged.wav"
DIR_SOURCE = "preprocessed_audio/"
DIR_TARGET = "dcapswoz_audio_participantonly_merged/"

def retrieveIdsFromCsv(filename):
	ids = []
	with open(filename, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			token = row[0]
			if token.lower() == COLUMN_HEADER:
				continue
			id = token + ID_EXTENSION
			ids.append(id)
	return ids


def predict(model,path):
    return random.choice([0,0,0,1,1,1,1,1,1,1])
			
def main():
	for partition in PARTITIONS:
		ids = retrieveIdsFromCsv(partition + PARTITION_EXTENSION)
		for id in ids:
			shutil.copy2(DIR_SOURCE + id, DIR_TARGET + partition + "/")
		

if __name__ == "__main__": 
	main()