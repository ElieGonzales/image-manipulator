#THE ULTIMATE IMAGE MANIPULATOR
import random
from PIL import Image, ImageTk, ImageOps
import numpy as np
import os
import colorsys
if __name__ == "__main__":
    import ttkbootstrap as ttk
    from tkinter import filedialog
    from tkinter.messagebox import askyesno, showinfo, showerror


#Creates an array of random greyscale values
def randomNoiseArrayScale(sizex, sizey):
    noise = []
    for _ in range(sizey):
        noise = []
        for _ in range(sizex):
            randomcol = random.randint(0, 255)
            noise.append([randomcol, randomcol, randomcol])
        noise.append(noise)
    return noise

#create an array of random black and white values
def randomNoiseArrayBool(sizex, sizey):
    noise = []
    for _ in range(sizey):
        noise = []
        for _ in range(sizex):
            randomcol = random.randint(0, 1)*255
            noise.append([randomcol, randomcol, randomcol])
        noise.append(noise)
    return noise

#creates an array of random colors
def randomNoiseArrayColor(sizex, sizey):
    noise = []
    for _ in range(sizey):
        row = []
        for _ in range(sizex):
            randomcol = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            row.append(randomcol)
        noise.append(noise)
    return noise

#Converts an array to an image
def arrayToImage(array):
    img = Image.new("RGB", (len(array[0]), len(array)))
    for y in range(len(array)):
        for x in range(len(array[0])):
            try:
                img.putpixel((x, y), tuple(array[y][x]))
            except:
                img.putpixel((x, y), int(array[y][x]))
    return img
    
def imageToArray(image):
    array = np.array(image)
    return array

#Creates a Noise Filter to the input image, with a random offset
def similarImageGen(img, offset=1):
    noise = []
    for y in range(img.size[1]):
        noise.append([])
        for x in range(img.size[0]):
            
            col = img.getpixel((x, y))
            if col[0] == col[1] == col[2]:
                coloffset = random.randint(-offset, offset ) 
                col1 = col[0]+coloffset
                col2 = col[1]+coloffset
                col3 = col[2]+coloffset
            else:
                col1 = col[0]+random.randint(-offset, offset)
                col2 = col[1]+random.randint(-offset, offset)
                col3 = col[2]+random.randint(-offset, offset)
            
            randomcol = [max(min(i, 255), 0) for i in [col1, col2, col3]]
            
            noise[y].append(randomcol)
    img = Image.fromarray(np.array(noise, dtype=np.uint8))
    return img 
  
#Creates a gif of noise with slight variations starting from the input image    
def smoothNoiseVideo(img, frames=20, offset=10, filename="noise.gif"):
    imgs = []
    for i in range(frames-1):
        simimg = similarImageGen(img, offset)
        simimg.save(str(i)+".png")
        imgs.append(simimg)
    img.save(filename, append_images=imgs, save_all=True, loop=0)   
    for i in range(frames-1):
        os.remove(str(i)+".png")

#makes a glitch effect on an image by flipping it's width and height but keeping the same pixels
def glitchEffect(img):
    array = np.array(img)    
    array = array.reshape((len(array[0]), len(array), 3))        
    return Image.fromarray(array)

#Takes two images and returns an image with the average of the two pixels
def imageAverageArray(image1, image2):
    imgx, imgy = image1.size
    
    pixels1 = image1.load()
    pixels2 = image2.load()
    
    result = []
    for y in range(imgy):
        row = []
        for x in range(imgx):
            row.append([(pixels1[x,y][0]+pixels2[x,y][0])//2, (pixels1[x,y][1]+pixels2[x,y][1])//2, (pixels1[x,y][2]+pixels2[x,y][2])//2])
        result.append(row)
    return result

#Makes an image symmetrical by mirroring it twice
def makeSymmetrical(img):
    newimg = Image.new("RGB", (img.width * 2, img.height * 2))
    newimg.paste(img, (0, 0))
    newimg.paste(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT), (img.width, 0))
    newimg.paste(img.transpose(Image.Transpose.FLIP_TOP_BOTTOM), (0, img.height))
    newimg.paste(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.FLIP_TOP_BOTTOM), (img.width, img.height))
    return newimg

#Generates a "procedural" image map that bases each pixel on the pixels around it
def generateImageMap(size):
    scale = size // 10
    array = []
    for i in range(size):
        row = []
        for i in range(size):
            #first pixel is truly random
            if len(row) == 0 and len(array) == 0:
                randomcol = random.randint(0, 255)
            #if it's the first pixel of the row, base it on the pixel above
            elif len(row) == 0:
                randomcol = array[-1][0][0] + random.randint(-scale, scale)
            #if it's the first row, base it on the pixel to the left
            elif len(array) == 0:
                randomcol = row[-1][0] + random.randint(-scale, scale)
            #else, base it on the pixel above and to the left
            else:
                randomcol = (array[-1][len(row)][0] + row[-1][0]) // 2 + random.randint(-scale, scale)
            #make sure it's not out of bounds
            if randomcol < 0:
                randomcol = 0
            elif randomcol > 255:
                randomcol = 255
            #append the pixel to the row
            row.append([randomcol, randomcol, randomcol])
        #append the row to the array
        array.append(row)
    img = Image.fromarray(np.array(array, dtype=np.uint8))
    return img

#Generates a "procedural" image map that bases each pixel on the pixels around it
def generateColorImageMap(size=1024, scale=20):
    noise = []
    for _ in range(size):
        row = []
        for _ in range(size):
            #first pixel is truly random
            if len(row) == 0 and len(noise) == 0:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
            #if it's the first pixel of the row, base it on the pixel above
            elif len(row) == 0:
                r = noise[-1][0][0] + random.randint(-scale, scale)
                g = noise[-1][0][1] + random.randint(-scale, scale)
                b = noise[-1][0][2] + random.randint(-scale, scale)
            #if it's the first row, base it on the pixel to the left
            elif len(noise) == 0:
                r = row[-1][0] + random.randint(-scale, scale)
                g = row[-1][1] + random.randint(-scale, scale)
                b = row[-1][2] + random.randint(-scale, scale)
            #else, base it on the pixel above and to the left
            else:
                r = (noise[-1][len(row)][0] + row[-1][0]) // 2 + random.randint(-scale, scale)
                g = (noise[-1][len(row)][1] + row[-1][1]) // 2 + random.randint(-scale, scale)
                b = (noise[-1][len(row)][2] + row[-1][2]) // 2 + random.randint(-scale, scale)
            
            #make sure it's not out of bounds (sorry for the disgusting code)
            r = 0 if r < 0 else 255 if r > 255 else r
            g = 0 if g < 0 else 255 if g > 255 else g
            b = 0 if b < 0 else 255 if b > 255 else b
            #appends the color to the row
            row.append([r, g, b])
        
        #appends the row to the noise
        noise.append(row)
    #makes sure the noise is the right size
    noise = noise[-size:]
    image = Image.fromarray(np.array(noise, dtype=np.uint8))
    return image

#Converts an array to an array of only black and white pixels
def arrayToBlackAndWhiteArray(initarray):
    finalarray = []
    for y in range(len(initarray)):
        row = []
        for x in range(len(initarray[y])):
            if initarray[y][x][0] > 127 and initarray[y][x][1] > 127 and initarray[y][x][2] > 127:
                row.append([255, 255, 255])
            else:
                row.append([0, 0, 0])
        finalarray.append(row)
    return finalarray

#Sorts the pixels of an image
def pixelSort(image, lowtreshold=50, hightreshold=200, sortmethod="brightness", showmask=False, direction="horizontal", sortorder=False):
    
    if sortorder == "ascending":
        reverseorder = False
    else:
        reverseorder = True
    
    def sortcolumn(column, sortmethod="brightness"):
        if sortmethod == "brightness":        
            column.sort(key=lambda rgb: sum(rgb)/3)
        elif sortmethod == "red":
            column.sort(key=lambda rgb: rgb[0])
        elif sortmethod == "green":
            column.sort(key=lambda rgb: rgb[1])
        elif sortmethod == "blue":
            column.sort(key=lambda rgb: rgb[2])
        elif sortmethod == "hue":
            column.sort(key=lambda rgb: colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])[0])
        elif sortmethod == "saturation":
            column.sort(key=lambda rgb: colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])[1])
        elif sortmethod == "value":
            column.sort(key=lambda rgb: colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])[2])
        
        return column

    origpixels = image.load()
    width, height = image.size

    mask = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            r, g, b = origpixels[x, y]
            if not (lowtreshold < (r+g+b)/3 < hightreshold):
                mask.putpixel((x, y), (0, 0, 0))
            else:
                mask.putpixel((x, y), (255, 255, 255))
    if showmask:
        mask.show()

    if direction == "horizontal":
        outerLoopRange = height
        innerLoopRange = width
        boundary = width - 1
    else:  # vertical
        outerLoopRange = width
        innerLoopRange = height
        boundary = height - 1

    for outer in range(outerLoopRange):
        column = []
        for inner in range(innerLoopRange):
            if direction == "horizontal":
                current_pixel = origpixels[inner, outer]
                mask_pixel = mask.getpixel((inner, outer))
            else:
                current_pixel = origpixels[outer, inner]
                mask_pixel = mask.getpixel((outer, inner))
            
            if mask_pixel == (255, 255, 255):
                column.append(current_pixel)
            else:
                if len(column) > 1:
                    column = sortcolumn(column, sortmethod)
                    if reverseorder:
                        column.reverse()
                    for i in range(len(column)):
                        if direction == "horizontal":
                            origpixels[inner - i - 1, outer] = column[i]
                        else:
                            origpixels[outer, inner - i - 1] = column[i]
                column = []

            if inner == boundary and len(column) > 1:
                column = sortcolumn(column, sortmethod)
                if reverseorder:
                    column.reverse()
                for i in range(len(column)):
                    if direction == "horizontal":
                        origpixels[inner - i, outer] = column[i]
                    else:
                        origpixels[outer, inner - i] = column[i]

    return Image.fromarray(np.asarray(image))

def generateImageFromText(imgmap = ["#  #","    ", "#  #"," ## "]):
    img = []
    for y in range(len(imgmap)):
        row = []
        for x in range(len(imgmap[y])):
            if imgmap[y][x] != " ":
                row.append([0, 0, 0])
            else:
                row.append([255, 255, 255])
        if len(row) != len(imgmap[0]):
            raise Exception("Invalid image map size")
        img.append(row)
    return Image.fromarray(np.array(img, dtype=np.uint8))


#PIXELISATION FUNCTIONS



#pixelizes an image by taking a pixel every PIXELSTEP pixels
def sampleColorPixelize(image, PIXELSTEP):
    #Sets up the lists
    pixelrow = []
    pixelarray = []
    
    #Gets the image size
    imgx, imgy = image.size
    
    #Loads the image into pixels
    image = image.load()
    
    #make sure PIXELSTEP is a common divisor of the image size
    if imgx % PIXELSTEP == 0 and imgy % PIXELSTEP == 0:
        for y in range(0, imgy, PIXELSTEP):
            for x in range(0, imgx, PIXELSTEP):
                #append every pixel out of PIXELSTEP to a row
                pixelrow.append(list(image[x,y]))
            
            #append the row to the final array
            pixelarray.append(pixelrow)
            #resets the row
            pixelrow = []
        
        #transforms the array into a numpy array
        image = np.array(pixelarray, dtype=np.uint8)
        #reshapes the array to fit the image shape
        image = np.reshape(image, (imgy // PIXELSTEP, imgx // PIXELSTEP, 3))
        #transforms the array into an image
        finalimage = Image.fromarray(image)
        return finalimage
    
    else:
        print("PIXELSTEP must be a common divisor of the image size")              
        
#built in pixelization function        
def builtInPixelize(image, PIXELSTEP):
    #gets the image size
    imgx, imgy = image.size
    #resizes the image
    finalimage = image.resize((imgx // PIXELSTEP, imgy // PIXELSTEP))
    return finalimage

#gets the average color of a pixel block
def averagePixelColorPixelize(image, PIXELSTEP):
    img_x, img_y = image.size
    image = np.array(image)
    pixel_array = []
    for y in range(0, img_y, PIXELSTEP):
        pixel_row = []
        for x in range(0, img_x, PIXELSTEP):
            block = image[y:y+PIXELSTEP, x:x+PIXELSTEP]
            avg_color = block.mean(axis=(0,1)).astype(np.uint8)
            pixel_row.append(avg_color)
        pixel_array.append(pixel_row)
    image = np.array(pixel_array, dtype=np.uint8)
    final_image = Image.fromarray(image)
    return final_image





if __name__ == "__main__":
    #Here's a list of all the functions and what they do
    #randomNoiseArrayScale(sizex, sizey) -> returns an array of random greyscale pixels
    #randomNoiseArrayBool(sizex, sizey) -> returns an array of random black and white pixels
    #randomNoiseArrayColor(sizex, sizey) -> returns an array of random color pixels
    #arrayToImage(array) -> returns an image from an array
    #imageToArray(image) -> returns an array from an image
    #similarImageGen(image, offset) -> returns an image that is similar to the original image, except with a random offset between -offset and offset
    #smoothNoiseVideo(img, frames, offset, filename) -> creates a video of shifting noise starting from img
    #glitchEffect(img) -> returns an image that looks glitched
    #imageAverageArray(img, img2) -> returns an array of the average color of each pixel of img and img2
    #makeSymmetrical(img) -> returns an image that is symmetrical by mirroring the image twice
    #generateImageMap(size) -> returns an array of pixels that are based on the pixels around them
    #generateColorImageMap(size, scale) -> returns an array of pixels that are based on the pixels around them, but in color
    #arrayToBlackandWhiteArray(array) -> returns an array of only black and white pixels
    #pixelSort(image, sortmethod, lowtreshold, hightreshold, showmask) -> returns an image that is pixel sorted
    #generateImageFromText(imgmap) -> returns a black and white image from an array of characters
    #sampleColorPixelize(image, PIXELSTEP) -> pixelizes an image by taking a pixel every PIXELSTEP pixels
    #builtInPixelize(image, PIXELSTEP) -> built in pixelization function
    #averagePixelColorPixelize(image, PIXELSTEP) -> gets the average color of a pixel block

    #TODO: Make undo button work
    #TODO: Make undo button work multiple times in a row (saving all iterations)
    #TODO: Make sliders more easily adjustable (see numbers or enter amount)
    #TODO: Polish UI/Add more filters
    
    
    root = ttk.Window(themename="cosmo")
    root.title("Image Processing")
    root.geometry("1480x720")
    root.resizable(True, True)
    
    
    WIDTH = 1280
    HEIGHT = 720
    file_path = ""
    
    left_frame = ttk.Frame(root, width=200, height=600)
    left_frame.pack(side="left", fill="y")
    
    canvas = ttk.Canvas(root, width=WIDTH, height=HEIGHT)
    canvas.pack()
    
    #add buttons for each function to the left frame
    filter_label = ttk.Label(left_frame, text="Select Filter:")
    filter_label.pack(padx=0, pady=2)
    
    image_filters = ["Noise Filter", "Glitch Effect", "Pixel Sort", "Pixelize", "Symmetrical", "Black and White", "Blur"]
    filter_combobox = ttk.Combobox(left_frame, values=image_filters, width=20)
    filter_combobox.pack(padx=10, pady=5)
    
    image_generation_label = ttk.Label(left_frame, text="Select Image Generation:")
    image_generation_label.pack(padx=0, pady=2)
    
    image_generation = ["Random Noise Greyscale", "Random Noise Black and White", "Random Noise Color", "Image Map", "Image Map Color"]
    generation_combobox = ttk.Combobox(left_frame, values=image_generation, width=20)
    generation_combobox.pack(padx=10, pady=5)
       
    
    def open_image():
        
        global file_path
        
        file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")])
        
        if file_path:
            global image, image_tk
            
            image = Image.open(file_path)
            image = image.convert("RGB")
            
            #new_width = int((WIDTH / 2))
            imagefortk = image.resize((WIDTH, HEIGHT), Image.LANCZOS)
                
            image_tk = ImageTk.PhotoImage(imagefortk)
            canvas.create_image(0, 0, anchor="nw", image=image_tk)
    
    def apply_filter(filter, options):
        
        global image, image_tk, last_used, prev_image
        prev_image = image
                
        #if part of options are empty, throw an error
        for key in options:
            if options[key] == "":
                showerror(title="Error", message="Please fill in all the options")
                return
        


        if filter == "Noise Filter":
            image = similarImageGen(image, options["noise_amount"])
        elif filter == "Glitch Effect":
            image = glitchEffect(image)
        elif filter == "Pixel Sort":
            doit = askyesno(title="Pixel Sort", message="This filter may take a while to process. Do you wish to continue?")
            if not doit:
                return
            image = pixelSort(image, sortmethod=options["sort_method"], lowtreshold=options["low_treshold"], hightreshold=options["high_treshold"], direction=options["direction"], sortorder=options["sort_order"])
        elif filter == "Pixelize":
            image = sampleColorPixelize(image, options["pixel_size"])
        elif filter == "Symmetrical":
            image = makeSymmetrical(image)
        elif filter == "Black and White":
            image = ImageOps.grayscale(image)
            image = image.convert("RGB")
        elif filter == "Blur":
            image = builtInPixelize(image, options["blur_amount"])
            
        last_used = filter
        
        imagefortk = image.resize((WIDTH, HEIGHT), Image.LANCZOS)
        image_tk = ImageTk.PhotoImage(imagefortk)
        canvas.create_image(0, 0, anchor="nw", image=image_tk)
    
    def generate_image(method):
        
        global image, image_tk, last_used, prev_image
        
        doit = askyesno(title="Generate Image", message="Are you sure you want to generate a new image? This will overwrite the current image.")
        
        if not doit:
            return
    
        prev_image = image
        
        if method == "Random Noise Greyscale":
            image = arrayToImage(randomNoiseArrayScale(720, 720))
        elif method == "Random Noise Black and White":
            image = arrayToImage(randomNoiseArrayBool(720, 720))
        elif method == "Random Noise Color":
            image = arrayToImage(randomNoiseArrayColor(720, 720))
        elif method == "Image Map":
            image = generateImageMap(720)
        elif method == "Image Map Color":
            image = generateColorImageMap(720, 10)
        
        last_used = method       
        
        image_tk = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=image_tk)
            
    def redo():
        global image, image_tk, last_used
    
        if last_used in image_filters:
            apply_filter(last_used)
        elif last_used in image_generation:
            generate_image(last_used)
    
    def undo():
        global image, image_tk, prev_image
        
        image = prev_image
        
        imagefortk = image.resize((WIDTH, HEIGHT), Image.LANCZOS)
        image_tk = ImageTk.PhotoImage(imagefortk)
        canvas.create_image(0, 0, anchor="nw", image=image_tk)
       
    def save_image():
        global image
        image.save("output.png")
        showinfo(title="Image Saved", message="Image saved as output.png")
        root.destroy()
            
    def show_filter_options(filter):
        
        
        
        if filter == "Noise Filter":
            noise_amount_label = ttk.Label(left_frame, text="Noise Amount:")
            noise_amount_label.pack(padx=0, pady=2)
            
            noise_amount_slider = ttk.Scale(left_frame, from_=0, to=100, orient="horizontal", length=200)
            noise_amount_slider.pack(padx=10, pady=5)
        elif filter == "Pixel Sort":
            sort_method_label = ttk.Label(left_frame, text="Sort Method:")
            sort_method_label.pack(padx=0, pady=2)
            
            sort_method_combobox = ttk.Combobox(left_frame, values=["Brightness", "Red", "Green", "Blue", "Hue", "Saturation", "Value"], width=20)
            sort_method_combobox.pack(padx=10, pady=5)
            
            
            low_treshold_label = ttk.Label(left_frame, text="Low Treshold:")
            low_treshold_label.pack(padx=0, pady=2)
            
            low_treshold_slider = ttk.Scale(left_frame, from_=-1, to=257, orient="horizontal", length=200)
            low_treshold_slider.pack(padx=10, pady=5)
            
            
            high_treshold_label = ttk.Label(left_frame, text="High Treshold:")
            high_treshold_label.pack(padx=0, pady=2)
            
            high_treshold_slider = ttk.Scale(left_frame, from_=-1, to=257, orient="horizontal", length=200)
            high_treshold_slider.pack(padx=10, pady=5)
            
            
            direction_label = ttk.Label(left_frame, text="Direction:")
            direction_label.pack(padx=0, pady=2)
            
            direction_combobox = ttk.Combobox(left_frame, values=["Vertical", "Horizontal"], width=20)
            direction_combobox.pack(padx=10, pady=5)
            
            
            sort_order_label = ttk.Label(left_frame, text="Sort Order:")
            sort_order_label.pack(padx=0, pady=2)
            
            sort_order_combobox = ttk.Combobox(left_frame, values=["Ascending", "Descending"], width=20)
            sort_order_combobox.pack(padx=10, pady=5)
            
        elif filter == "Pixelize":
            pixel_size_label = ttk.Label(left_frame, text="Pixel Size:")
            pixel_size_label.pack(padx=0, pady=2)
            
            pixel_size_slider = ttk.Scale(left_frame, from_=1, to=100, orient="horizontal", length=200)
            pixel_size_slider.pack(padx=10, pady=5)
        
        elif filter == "Blur":
            blur_amount_label = ttk.Label(left_frame, text="Blur Amount:")
            blur_amount_label.pack(padx=0, pady=2)
            
            blur_amount_slider = ttk.Scale(left_frame, from_=1, to=100, orient="horizontal", length=200)
            blur_amount_slider.pack(padx=10, pady=5)
            
        def apply():
            
            options = {}
            
            if filter == "Noise Filter":
                options["noise_amount"] = noise_amount_slider.get()
            elif filter == "Pixel Sort":
                options["sort_method"] = sort_method_combobox.get().lower()
                options["low_treshold"] = low_treshold_slider.get()
                options["high_treshold"] = high_treshold_slider.get()
                options["direction"] = direction_combobox.get().lower()
                options["sort_order"] = sort_order_combobox.get().lower()
            elif filter == "Pixelize":
                options["pixel_size"] = pixel_size_slider.get()
            elif filter == "Blur":
                options["blur_amount"] = blur_amount_slider.get() 
            
            apply_filter(filter, options)
            
            

        apply_button = ttk.Button(left_frame, bootstyle="dark", command=apply, text="Apply")
        apply_button.pack(pady=10)
    
    image_button = ttk.Button(left_frame, bootstyle="light", command=open_image, text="Open Image")
    image_button.pack(pady=5)
    
    save_button = ttk.Button(left_frame, bootstyle="light", command=save_image, text="Save Image")
    save_button.pack(pady=5)
    
    redo_button = ttk.Button(left_frame, bootstyle="light", command=redo, text="Apply Last Used")
    redo_button.pack(pady=5)
    
    undo_button = ttk.Button(left_frame, bootstyle="light", command=undo, text="Undo")
    undo_button.pack(pady=5)
    
    filter_combobox.bind("<<ComboboxSelected>>", lambda _: show_filter_options(filter_combobox.get()))
    generation_combobox.bind("<<ComboboxSelected>>", lambda _: generate_image(generation_combobox.get()))
            
    root.mainloop()
            
            
    
