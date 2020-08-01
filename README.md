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

`INPUT_IMG_NAME     = 'data/sonic.jpg'`
`RESIZE_TO          = 500`
`SLICE_SIZE         = 5`
`REBUILD_SLICE_SIZE = 100`
`ALPHA              = 0.42`
`OUTPUT_IMG_NAME    = 'sonic.jpg'`

`CPU                = 2`

`CODE_IMG_SIZE      = 100`
`IMG_EXTENSIONS     = ['jpg', 'png']`
`CODE_NAME          = 'code1'`
`CODE_IMG_DIR       = '/media/parthikb/2E9D987E5CF82669/Pics/Trip-Kausani'`


1. First, you need to build a `Code.npy` that will have the information of your Images(from which you'll create other images).
2. Change the Parameters in the `PARAMETERS.py` according to you.
2. 
