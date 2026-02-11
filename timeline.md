# codebase imgs

Old ckpt test, def some settings lost to orig, compression is in 3-5x range, not 10-13x

<img width="2549" height="1347" alt="image" src="https://github.com/user-attachments/assets/a101f7b7-c875-4060-8706-6a1ea5917360" />
<img width="2547" height="1357" alt="image" src="https://github.com/user-attachments/assets/b0a601f2-1af9-426e-999e-d0df48a6abda" />
<img width="2548" height="1367" alt="image" src="https://github.com/user-attachments/assets/3438d537-9322-4bc7-8512-b319201ee50f" />

within testing and datachecks, modern equivalent of the old checkpoint, back in the old range
the color compression should fix with training

<img width="2000" height="1000" alt="image" src="https://github.com/user-attachments/assets/a23a1ca5-c488-4c67-9e39-4dc2a9609e25" />

more data doesnt hurt SSIM loss, structure improves, but it forces the model to deal with a larger color spectrum, which increases loss, but helps model performance in a way, colors match better

<img width="2000" height="1000" alt="image" src="https://github.com/user-attachments/assets/1bbf0f78-2b98-41eb-a4a9-9db851500cc4" /><br>
> note, zfp was much weaker on this one, oddly enough compression seems to scale with loss or that color is more compressible when you decrease range?

<br><br>

<img width="562" height="432" alt="image" src="https://github.com/user-attachments/assets/d4e49c2c-d5ee-4205-ba06-001a62545080" /><br>
> better detail instantly ruins compressiblity
