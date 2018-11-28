pip3 install -r requirements.txt

if [[ "$OSTYPE" == "darwin"* ]]; then
	# Detect osx version
	os_ver=${1-:$(sw_vers -productVersion)}
	if [[ "$os_ver" == :10.14.* ]]; then
		# thanks to https://github.com/ContinuumIO/anaconda-issues/issues/10135
		export MACOSX_DEPLOYMENT_TARGET=10.14
	fi
fi
python3 setup.py build_ext --inplace