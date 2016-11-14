# let's do 4 subject, 6 gestures (1 ~ 6) each

rm -rf "processed_full_subtract_only"
rm -rf "processed_merged"
rm -rf "cropped_merged"
rm -rf "cropped_final"

source ~/.bash_profile 

subject=1
while [ $subject -lt 6 ]; do
	if [ $subject -ne 2 ]; then
		gesture=1
		while [ $gesture -lt 7 ]; do
			echo "Preprocess for ${subject} ${gesture}"
			python preprocess.py $subject $gesture
			gesture=$((gesture+1))
		done
	fi
	subject=$((subject+1))
done

#We start with the processed image with subject/gsource ~/.bash_profile esture/imgs structure
python mergeGestureFolder.py --input "processed_full_subtract_only" --output "processed_merged"

#crop the images and renaming them -> output: imgs & txts
python handDetector.py  --input "processed_merged" --output "cropped_merged"  --step 10

#detect if the candidates are actually hands and merge
python mergeSlices.py 