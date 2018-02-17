#!/bin/bash
mkdir submission

if [ -e submission.zip ]
then
rm submission.zip
fi

if [ -e PorterStemmer.py ]
then
    cp PorterStemmer.py submission/
fi

if [ -d deps ]
then
    cp -R deps submission/
fi

cp chatbot.py submission/
cd submission
zip -r submission.zip *
mv submission.zip ../
cd ../
rm -r submission

