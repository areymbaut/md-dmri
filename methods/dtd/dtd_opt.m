function opt = dtd_opt(opt)
% function opt = dtd_opt(opt)
%
% Makes sure that all needed fields in the options structure are present

opt.dtd.present = 1;

opt.dtd = msf_ensure_field(opt.dtd, 'tmp', 1); 
opt.dtd = msf_ensure_field(opt.dtd, 'do_plot', 0);
opt.dtd = msf_ensure_field(opt.dtd, 'ind_start', 1);
opt.dtd = msf_ensure_field(opt.dtd, 'dmin', 1e-11);
opt.dtd = msf_ensure_field(opt.dtd, 'dmax', 5e-9);
opt.dtd = msf_ensure_field(opt.dtd, 'n_in', 2e2); % n_in: Number of nodes in NNLS inversion. [100 - 1000]
opt.dtd = msf_ensure_field(opt.dtd, 'n_out', 10);
opt.dtd = msf_ensure_field(opt.dtd, 'n_kill', 1);
opt.dtd = msf_ensure_field(opt.dtd, 'n_proliferation', 20);
opt.dtd = msf_ensure_field(opt.dtd, 'n_extinction', 20);
opt.dtd = msf_ensure_field(opt.dtd, 'ofuzz', .1*2*pi);
opt.dtd = msf_ensure_field(opt.dtd, 'dfuzz', .1);

opt.dtd = msf_ensure_field(opt.dtd, 'fig_maps', ...
    {'s0','miso','viso_n','msaniso_n','vsaniso_n'});
opt.dtd = msf_ensure_field(opt.dtd, 'fig_prefix', 'dtd');
opt.dtd = msf_ensure_field(opt.dtd, 'fig_cmaps',{'fa','cl','cp','ufa'});
opt.dtd = msf_ensure_field(opt.dtd, 'fig_ccol',{'t1x6','lambda33vec','lambda11vec','s1x6prim'});
opt.dtd = msf_ensure_field(opt.dtd, 'fig_ccolnorm',{'lambda33','mask','mask','slambda33prim'});
