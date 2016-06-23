function EG = mgui_browse_setup_gui(EG, h_panel)
% function EG = mgui_browse_setup_gui(EG, h_panel)

h       = [0.999 0.001];
p_left  = 0.03;
w       = 1 - 2 * p_left;
mh      = 0.01;
b       = 0.02;
h       = h / sum(h) * (1 - 2*b - numel(h) * mh);
h       = h * 0.98;

uicontrol(...
    'Style','listbox',...
    'FontSize', EG.conf.default_font_size, ...
    'Parent', h_panel, ...
    'String','No ROI selected',...
    'Interruptible', 'off', ...
    'Units', 'Normalized', ...
    'Position',[p_left b w h(2)], ...
    'Tag', EG.t_BROWSE_ROI, ...
    'Callback', EG.f_callback, ...
    'Visible', 'off');

b = b + h(2) + mh;

uicontrol(...
    'Style','listbox',...
    'FontSize', EG.conf.default_font_size, ...
    'Parent', h_panel, ...
    'String','-',...
    'Interruptible', 'off', ...
    'Units', 'Normalized', ...
    'Position',[p_left b w h(1)], ...
    'Tag', EG.t_BROWSE_FILE, ...
    'Callback', EG.f_callback);


b = b + h(1) + mh;
h_button = 0.028;
lw = 0.6;

uicontrol(...
    'Style','pushbutton',...
    'FontSize', EG.conf.default_font_size, ...
    'Parent', h_panel, ...
    'String','Open dir',...
    'Interruptible', 'off', ...
    'Units', 'Normalized', ...
    'Position',[p_left b w * lw h_button], ...
    'Tag', EG.t_BROWSE_FOLDER, ...
    'Callback', EG.f_callback);

uicontrol(...
    'Style','popupmenu',...
    'FontSize', EG.conf.default_font_size, ...
    'Parent', h_panel, ...
    'String', 'ext', ...
    'Interruptible', 'off', ...
    'Units', 'Normalized', ...
    'Position',[p_left + (lw+0.01) * w b w * (1-lw-0.01) h_button], ...
    'Tag', EG.t_BROWSE_EXT, ...
    'Callback', EG.f_callback);

end
