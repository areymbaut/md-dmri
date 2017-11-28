function bt = gwf_to_bt(gwf, rf, dt)
% function bt = gwf_to_bt(gwf, rf, dt)
%
% gwf - gradient waveform of size N x 3
% rf  - effect of rf pulses (range -1 to 1), size N x 1
% dt  - time step of waveform
%
% Following notation in Westin et al (2016) NeuroImage 135

gwf_check(gwf, rf, dt);

% integrating q^2 requires some additional resolution, in order avoid bias
[gwf,rf,dt] = gwf_interpolate(gwf, rf, dt, 16);

% compute q and the bt
q = gwf_to_q(gwf, rf, dt);

bt = q' * q * dt;

