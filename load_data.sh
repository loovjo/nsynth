wget "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz" -O "nsynth.tar.gz"
mkdir ~/nsynth_
mkdir ~/nsynth
tar -xvf nsynth.tar.gz -C ~/nsynth_
rm nsynth.tar.gz
mv ~/nsynth_/*/* ~/nsynth
rm -r ~/nsynth_