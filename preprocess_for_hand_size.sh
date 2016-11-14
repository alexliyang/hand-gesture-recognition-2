subject=0
while [ $subject -lt 6 ]; do
	if [ $subject -ne 2 ]; then
		gesture=1
		while [ $gesture -lt 7 ]; do
			echo "Process for ${subject} ${gesture}"
			python preprocess.py $subject $gesture
			gesture=$((gesture+1))
		done
	fi
	subject=$((subject+1))
done

python mergeGestureFolder.py --input "processed_full_subtract_only" --output "processed_merged"