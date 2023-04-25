load('MicTimeSeries.mat')
load('turbo.mat')

Fs = 25000;
%% Calculate spectrogram
T_interval_spect = 1;
Num_spect_Averaging = 2;
overlap_spect_percentage = 0.9;
Spect_PSD_len = floor(Fs*T_interval_spect);

window_spect = hann(Spect_PSD_len);

[s,freq_spec,t_spec,spect] = (spectrogram(detrend(Data_CAL), window_spect, ...
    floor(overlap_spect_percentage*Spect_PSD_len), ...
    Spect_PSD_len, ...
    Fs,'yaxis'));

%% Plot spectrogram

figure(1)
clf

limlow = -5.2;
limhigh = -2.9;

xlimlow = t_spec(1);
xlimhigh = t_spec(end);
ylimlow = 10;%
ylimhigh = 1000;
clut = turbo;
Ax = 'lin';
hold on
colormap(clut);
surf(t_spec, freq_spec, log10(sqrt(spect)), 'EdgeColor', 'none')
shading(gca, 'flat');
set(gca,'YScale',Ax);
caxis([limlow limhigh]);
set(gca, 'layer', 'top');
colorbar;
h = colorbar;

xlabel('Time s','FontSize',14);
xlim([xlimlow xlimhigh])

ylabel('Frequency (Hz)','FontSize',14);
ylim([ylimlow, ylimhigh])

ylabel(h, ' (log(Pa/\surdHZ))','FontSize',14)
box on
title('Sol 148')
