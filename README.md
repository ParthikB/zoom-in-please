# zoom in please!

*I really don't remember how this idea came to me and I didn't wasted much time, I wanted to get this done quickly 'cause I wanted to see the results! This is just so cool.*

So what's happening here? 

Here, I'm re-building an Image using many other smaller Images which are similar in color to the Pixel Block it's replaced with. 
Umm ...I don't think I did a great job explaining this, but well, why don't you **zoom-in-please**!

<h2>Input Image :</h2>

![](/data/sonic.jpg)

<h2>Output Image :</h2>

![](output.gif)


So how to do this? That's simple, just follow the following steps and you'll be good to go!

# PARAMETERS.PY:

`INPUT_IMG_NAME     = 'data/sonic.jpg'` # The Image you want to rebuild (with complete path and extension)

`RESIZE_TO          = 500` # Default works fine in most of the cases. Still, you can play around.

`SLICE_SIZE         = 5` # The size of each slice. Thus, there will be 500/5 = 100 slices. Thus, 100x100 slices in total.

`REBUILD_SLICE_SIZE = 100`# Leave Default

`ALPHA              = 0.42`# Control the transparency of the Original Image overlayed onto the final image. Leave Default.

`OUTPUT_IMG_NAME    = 'sonic.jpg'` # Name of the Output Image

`CPU                = 2` # Numbers of CPUs to use in multiprocessing.

`CODE_IMG_SIZE      = 100`# Leave Default

`IMG_EXTENSIONS     = ['jpg', 'png']` # The extensions you want to focus on while selecting the Images in bulk.

`CODE_NAME          = 'code1'`# Name of the Code file

`CODE_IMG_DIR       = '/media/parthikb/2E9D987E5CF82669/Pics/Trip-Kausani'` # The Directory in which the Code Images will be present.


1. First, you need to build a `Code.npy` that will have the information of your Images(from which you'll create other images).
2. Change the Parameters in the `PARAMETERS.py` according to you.
3. Run `python create_code.py`. This will create a Code file.
4. Run `python run.py` and let the program do its job. You relax!

Any suggestion? I'm always up for it!

Also, check out my website : <a href="https://www.parthiktalks.com"> parthiktalks.com </a>
