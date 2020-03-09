% ref = imread("11.jpg");
% %ref= rgb2gray(ref);
% %imshow(ref)
% 
%A = imnoise(ref,'gaussian',0,0.1);


I = imread("11_g_0.1.jpg");
I = rgb2gray(I);
A = imdiffusefilt(I);
%A = imnlmfilt(I);
%A = imgaussfilt(I,0.02);
% %A = wiener2(I,[]);
%A = imresize(A,[165 137]);
imshow(A)

% [peaksnr, snr] = psnr(A, ref);  
% fprintf('\n The Peak-SNR value is %0.4f', peaksnr);

% A = imread("31.png");
% se = strel('disk',9);
% E10 = imerode(A,se);
% imshow(E10);

% A = imread("Y2531_diffusionfckmean.jpg");
% se = strel('disk',3);
% E10 = imerode(A,se);
% imshow(E10);