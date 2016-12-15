#!/bin/bash

CURRENTVERSION=$(python -c 'import ionn; print(ionn.__version__)')
echo Current version: $CURRENTVERSION

read MAJOR MINOR FIX <<< $(echo $CURRENTVERSION | sed -e 's/\./ /g')
echo $MAJOR  $MINOR  $FIX

case $1 in
    major*)
        echo Performing major release;
        MAJOR=$(($MAJOR + 1));
        MINOR="0";
        FIX="0";
        ;;
    minor*)
        echo Performing minor release
        MINOR=$(($MINOR + 1));
        FIX="0";
        ;;
    fix*)
        echo Performing subrelease
        FIX=$(($FIX + 1));
        ;;
esac

echo $(cat ionn/__init__.py | sed -e "s/[0-9]*\.[0-9]*\.[0-9]*/$MAJOR.$MINOR.$FIX/") > ionn/__init__.py

# This would perform the actual release (if tests pass)
git commit ionn/__init__.py -m "Version bump"
git tag -a v$MAJOR.$MINOR.$FIX -m "$*"
git push --tags origin master
