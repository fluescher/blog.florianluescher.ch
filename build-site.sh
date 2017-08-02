#!/bin/bash

docker run \
	-it \
	-p 4000:4000 \
	--volume=$PWD:/srv/jekyll \
	jekyll/jekyll \
	jekyll serve --drafts
