for dim in 50 100 200 300 400
do
	release/DPmixGGM_SSS.exe lymph_best"$dim" 1 > RES/out_best"$dim"_MCMC1.txt &
	release/DPmixGGM_SSS.exe lymph_best"$dim" 2 > RES/out_best"$dim"_MCMC2.txt &
	release/DPmixGGM_SSS.exe lymph_best"$dim" 3 > RES/out_best"$dim"_MCMC3.txt
done