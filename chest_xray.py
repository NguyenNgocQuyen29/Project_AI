from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

model = load_model('Chest_XRAY_cnn.h5')

def CTCHINH():
    image_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if image_path:
        img = Image.open(image_path).resize((200, 200))
        plt.imshow(img)
        img_array = image_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        a = np.argmax(model.predict(img_array), axis=-1)

        root1 = Tk()
        root1.title("Results")
        root1.geometry("240x120")
        
        if a == 0:
            result_label = Label(root1, text="Dự đoán: Covid_19", font="Arial 13 ", fg="red")
            result_label.place(x=100, y=50, anchor=CENTER)
        elif a == 1:
            result_label = Label(root1, text="Dự đoán: Normal", font="Arial 13 ", fg="red")
            result_label.place(x=100, y=50, anchor=CENTER)
        elif a == 2:
            result_label = Label(root1, text="Dự đoán: Viral Pneumonia", font="Arial 13 ", fg="red")
            result_label.place(x=100, y=50, anchor=CENTER)
        elif a == 3:
            result_label = Label(root1, text="Dự đoán: Bacterial Pneumonia", font="Arial 13 ", fg="red")
            result_label.place(x=100, y=50, anchor=CENTER)
            
        plt.imshow(img.convert('L'), cmap='gray')  # Hiển thị ảnh gốc dưới dạng ảnh đen trắng
        plt.show()
        root1.mainloop()

def image_to_array(image):
    img_array = image.resize((200, 200))
    img_array = np.array(img_array)
    img_array = img_array.astype('float32')
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Mở rộng chiều cuối cùng
    img_array = np.repeat(img_array, 3, axis=-1)  # Lặp lại kênh màu 3 lần
    return img_array


# Tạo cửa sổ giao diện chính
root = Tk()
root.title("Program Interface")
root.geometry("1366x720")

lgtruong = ImageTk.PhotoImage(Image.open("logotruong.png"))
lblgtruong = Label(image= lgtruong)
lblgtruong.place(x = 20, y = 5)

lgkhoa = Image.open("logokhoa.png")
resize_down1 = lgkhoa.resize((112,105))
lgkhoa_re = ImageTk.PhotoImage(resize_down1)
lblgkhoa = Label(image= lgkhoa_re)
lblgkhoa.place(x = 1150, y = 60,anchor = CENTER)

myLabel2 = Label(root, text="Faculty Of Mechanical Engineering", font="Arial 15 bold", fg="darkblue")
myLabel3 = Label(root, text="Mechatronics Engineering", font="Arial 15 bold", fg="darkblue")
myLabel2.place(x = 1150, y = 130,anchor = CENTER)
myLabel3.place(x = 1150, y = 160,anchor = CENTER)

anh2 = Image.open("phoi.jpg")
resize_down2 = anh2.resize((400,400))
anh2_re = ImageTk.PhotoImage(resize_down2)
lbanh2 = Label(image= anh2_re)
lbanh2.place(x = 80, y = 270)

anh1 = Image.open("nn.jpg")
resize_down3 = anh1.resize((200,200))
anh1_re = ImageTk.PhotoImage(resize_down3)
lbanh1 = Label(image= anh1_re)
lbanh1.place(x = 550, y = 310)


myLabel4 = Label(root, text="NHẬN DIỆN VÀ PHÂN LOẠI BỆNH VỀ LỒNG NGỰC BẰNG ẢNH X_RAY", font="Arial 23 bold", fg="red")
myLabel4.place(x = 683, y = 225,anchor = CENTER)

myLabel5 = Label(root, text="Artificial Intelligence", font="Arial 20 bold")
myLabel6 = Label(root, text="Instructors:  PGS. Nguyen Truong Thinh", font="Arial 15 bold")
myLabel5.place(x = 700, y = 175,anchor = CENTER)
myLabel6.place(x = 995, y = 400,anchor = CENTER)

myLabel7 = Label(root, text="MSSV:          20146523", font="Arial 15 bold", fg="black")
myLabel8 = Label(root, text="Author:         Nguyen Thi Ngoc Quyen", font="Arial 15 bold")

myLabel7.place(x = 800, y = 475)
myLabel8.place(x = 800, y =430)

btnLaunch = Button(root,text="TEST",font="Arial 30 bold", fg="red", command=CTCHINH)
btnLaunch.place(x = 650, y =600, anchor = CENTER)
btnStop = Button(root,text="QUIT",font="Arial 30 bold", fg="red", command=root.destroy)
btnStop.place(x = 1150, y =600, anchor = CENTER)
# Gọi vòng lặp sự kiện chính để các hành động có thể diễn ra trên màn hình máy tính của người dùng
root.mainloop()




