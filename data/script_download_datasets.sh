

# Command to download dataset:
#   bash script_download_datasets.sh

DIR=datasets/

if [ ! -d "$DIR" ] ; then
    mkdir $DIR
fi

cd $DIR

FILE=BACE.csv
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/o8we1vbf6gionq0/BACE.csv?dl=1 -o BACE.csv -J -L -k
fi

FILE=BBBP.csv
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/2en1xinxnvh7egg/BBBP.csv?dl=1 -o BBBP.csv -J -L -k
fi

FILE=HIV.csv
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/u1qhn94i2xrum5e/HIV.csv?dl=1 -o HIV.csv -J -L -k
fi

FILE=tox21_mnet.csv
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/8dq9z423n472lzb/tox21_mnet.csv?dl=1 -o tox21_mnet.csv -J -L -k
fi

cd ..

