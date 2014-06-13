for dim in 50 100 200 300 400
do
	release/DPmixGGM_SSS.exe lymph_best"$dim" 3 > RES/out_best"$dim"_SSS_GPU.txt
done