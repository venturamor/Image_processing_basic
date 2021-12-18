% 1 - info from freq
close all; clear all;
%% section 1.1
%addpath('C:\Matlab\bin\Anat_ex1');
%load('Diego.jpg');
img = imread('.\Diego.jpg');
figure(1);subplot(1,2,1); imshow(img); %original img
title('Original Image');
img_fft = fftshift(fft2(img));
subplot(1,2,2);imshow(log(1+abs(img_fft)),[]); %fft img
title('log(1+abs(FFT Image))');
impixelinfo;
%% section 1.2
mask = zeros(size(img_fft));
five_precent_size = floor(size(mask,1)*0.05);
mask(size(mask,1)/2-five_precent_size/2:size(mask,1)/2+five_precent_size/2,:)= 1;
mask(:,size(mask,1)/2-five_precent_size/2:size(mask,1)/2+five_precent_size/2)= 1;
figure(); imshow(mask,[]); %mask
img_fft_w_mask = mask.*img_fft;
figure(3);subplot(1,2,1); imshow(log(1+abs(img_fft_w_mask)),[]);%output fft
title('log(1+abs(FFT Image with Mask LPF)')
img_w_mask = ifft2(ifftshift(img_fft_w_mask));
subplot(1,2,2); imshow(img_w_mask,[]);%img with mask in space dimension
title('Image with Mask LPF')

%% section 1.3
col_sums = sum(img_fft);
[col_sums_sorted,col_ind] = sort(col_sums);
max_col_sums = col_sums_sorted(end-five_precent_size+1:end);
max_col_ind = col_ind(end-five_precent_size+1:end);

%% section 1.4
row_sums = sum(img_fft,2)';
[row_sums_sorted,row_ind] = sort(row_sums);
max_row_sums = row_sums_sorted(end-five_precent_size+1:end);
max_row_ind = row_ind(end-five_precent_size+1:end);

%% section 1.5
mask2 = zeros(size(img_fft));
mask2(:,max_col_ind) = 1;
mask2(max_row_ind,:) = 1;
figure(); imshow(mask2);impixelinfo;
img_fft_w_mask2 = mask2.*img_fft;
figure(5); subplot(1,2,1);imshow(log(1+abs(img_fft_w_mask2)),[]);%output fft
title('log(1+abs(FFT Image with Mask2)');
img_w_mask2 = ifft2(ifftshift(img_fft_w_mask2));
subplot(1,2,2); imshow(img_w_mask2,[]);%img with mask2 in space dimension
title('Image with Mask2');
%% section 1.6
ten_precent_size = floor((size(mask,1)^2)*0.1);
[max_mat, vec_mat_ind] = maxk(abs(img_fft(:)),ten_precent_size);
[col_mat,row_mat] = ind2sub(size(img_fft),vec_mat_ind);
mask3 = zeros(size(img_fft,1)*size(img_fft,1),1);
mask3(vec_mat_ind) = 1;
mask3 = reshape(mask3,[size(mask,1),size(mask,2)]);
figure(); imshow(mask3,[]);impixelinfo;
img_fft_w_mask3 = mask3.*img_fft;
figure(7); subplot(1,2,1); imshow(log(1+abs(img_fft_w_mask3)),[]);%output fft
title('log(1+abs(FFT Image with Mask3)');
img_w_mask3 = ifft2(ifftshift(img_fft_w_mask3));
subplot(1,2,2); imshow(img_w_mask3,[]);%img with mask3 in space dimension
title('Image with Mask3');
% 2 - phase and amplitude
%% section 2.1
clear all; close all; clc;
%addpath('C:\Matlab\bin\Anat_ex1');
img_dog = imread('.\dog.jpg');
img_yours = rgb2gray(imread('.\yours.jpg'));
img_yours = imresize(img_yours, [512,512]);% already checked 96 dpi :)
figure();subplot(1,3,1);
imshow(img_yours);title('Original Image');
subplot(1,3,2);
img_yours_fft = fft2(img_yours);
amp_yours = abs(img_yours_fft);
imshow(log(1+fftshift(amp_yours)),[]);
title('log(1+fftshift(FFT Amplitude Image))');
subplot(1,3,3);
phase_yours = angle(img_yours_fft);
imshow(phase_yours);title('FFT Phase Image');
figure();
subplot(1,3,1);
imshow(img_dog);title('Original Image');
subplot(1,3,2);
img_dog_fft = fft2(img_dog);
amp_dog = abs(img_dog_fft);
imshow(log(1+fftshift(amp_dog)),[]);
title('log(1+fftshift(FFT Amplitude Image))');
subplot(1,3,3);
phase_dog = angle(img_dog_fft);
imshow(phase_dog);
title('FFT Phase Image');
%% section 2.2
ampYours_phaseDog = ifft2(ifftshift(amp_yours.*(exp(1i*phase_dog))));
ampDog_phaseYours = ifft2(ifftshift(amp_dog.*(exp(1i*phase_yours))));
figure(); 
imshow(ampYours_phaseDog,[]);
title('AmpYours PhaseDog');
figure();
imshow(ampDog_phaseYours,[]);
title('AmpDog PhaseYours');
%% section 2.3
max_a = max(amp_yours,[],'all');
amp_rand = rand(size(amp_yours))*max_a;
phase_rand = pi*(rand(size(phase_dog))*2-1);
ampRand_phaseYours = ifft2(ifftshift(amp_rand.*(exp(1i*phase_yours))));
ampYours_phaseRand = ifft2(ifftshift(amp_dog.*(exp(1i*phase_rand))));
figure();imshow(ampRand_phaseYours,[]);
title('AmpRand PhaseYours');
figure();imshow(ampYours_phaseRand,[]);
title('AmpYours PhaseRand');

%3 - dot action in time
%% section 3.2
clear all; close all; clc;
%addpath('C:\Matlab\bin\Anat_ex1');
vid_corsica = VideoReader('.\Corsica.mp4');
% first time-session (a)
vid_corsica.CurrentTime = 1*60+15; %min to sec
currAxes = axes;
i=1;
while vid_corsica.CurrentTime < (1*60+16)
    vidFrame = readFrame(vid_corsica);
    vidFrames1(:,:,:,i) = vidFrame;
    image(vidFrame, 'Parent', currAxes);
    currAxes.Visible = 'off';
    pause(1/vid_corsica.FrameRate);
    i = i+1;
end
% second time-session (b)
vid_corsica.CurrentTime = 2*60+20; %min to sec
currAxes1 = axes;
i=1;
while vid_corsica.CurrentTime < (2*60+21)
    vidFrame = readFrame(vid_corsica);
    vidFrames2(:,:,:,i) = vidFrame;
    image(vidFrame, 'Parent', currAxes1);
    currAxes1.Visible = 'off';
    pause(1/vid_corsica.FrameRate);
    i = i+1;
end

%% section 3.3
% first time-session (a)
avg_frame1 = uint8(mean(vidFrames1,4));
med_frame1 = median(vidFrames1,4);
min_frame1 = min(vidFrames1,[],4);
max_frame1 = max(vidFrames1,[],4);

figure();%Frames (a)
subplot(2,2,1); imshow(avg_frame1,[]);title('Average Frame');
subplot(2,2,2); imshow(med_frame1,[]);title('Median Frame');
subplot(2,2,3); imshow(min_frame1,[]);title('Min Frame');
subplot(2,2,4); imshow(max_frame1,[]);title('Max Frame');

% second time-session (b)
avg_frame2 = uint8(mean(vidFrames2,4));
med_frame2 = median(vidFrames2,4);
min_frame2 = min(vidFrames2,[],4);
max_frame2 = max(vidFrames2,[],4);

figure();%Frames (b)
subplot(2,2,1); imshow(avg_frame2,[]);title('Average Frame');
subplot(2,2,2); imshow(med_frame2,[]);title('Median Frame');
subplot(2,2,3); imshow(min_frame2,[]);title('Min Frame');
subplot(2,2,4); imshow(max_frame2,[]);title('Max Frame');

%% section 3.4
% 3 seconds sessions
% first time-session (a)
vid_corsica.CurrentTime = 1*60+13; % min to sec
currAxes = axes;
i=1;
while vid_corsica.CurrentTime < (1*60+16)
    vidFrame = readFrame(vid_corsica);
    vidFrames3(:,:,:,i) = vidFrame;
    image(vidFrame, 'Parent', currAxes);
    currAxes.Visible = 'off';
    pause(1/vid_corsica.FrameRate);
    i = i+1;
end
% second time-session (b)
vid_corsica.CurrentTime = 2*60+20; %min to sec
currAxes1 = axes;
i=1;
while vid_corsica.CurrentTime < (2*60+23)
    vidFrame = readFrame(vid_corsica);
    vidFrames4(:,:,:,i) = vidFrame;
    image(vidFrame, 'Parent', currAxes1);
    currAxes1.Visible = 'off';
    pause(1/vid_corsica.FrameRate);
    i = i+1;
end

% first long time-session (a)
avg_frame3 = uint8(mean(vidFrames3,4));
med_frame3 = median(vidFrames3,4);
min_frame3 = min(vidFrames3,[],4);
max_frame3 = max(vidFrames3,[],4);

figure();%Frames (a)
subplot(2,2,1); imshow(avg_frame3,[]);title('Average Frame');
subplot(2,2,2); imshow(med_frame3,[]);title('Median Frame');
subplot(2,2,3); imshow(min_frame3,[]);title('Min Frame');
subplot(2,2,4); imshow(max_frame3,[]);title('Max Frame');

% second long time-session (b)
avg_frame4 = uint8(mean(vidFrames4,4));
med_frame4 = median(vidFrames4,4);
min_frame4 = min(vidFrames4,[],4);
max_frame4 = max(vidFrames4,[],4);

figure();%Frames (b)
subplot(2,2,1); imshow(avg_frame4,[]);title('Average Frame');
subplot(2,2,2); imshow(med_frame4,[]);title('Median Frame');
subplot(2,2,3); imshow(min_frame4,[]);title('Min Frame');
subplot(2,2,4); imshow(max_frame4,[]);title('Max Frame');
