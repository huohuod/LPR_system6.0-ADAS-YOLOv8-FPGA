onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/clk
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/r_in
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/g_in
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/b_in
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/vs_in
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/hs_in
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/de_in
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/video_enhance_lightdown_num
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/video_enhance_lightdown_sw
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/video_enhance_darkup_num
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/video_enhance_darkup_sw
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/y_out
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/u_out
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/v_out
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/vs_out
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/hs_out
add wave -noupdate -expand -label sim:/tb_rgb2yuv/Group1 -group {Region: sim:/tb_rgb2yuv} /tb_rgb2yuv/de_out
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {157587 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 150
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ps
update
WaveRestoreZoom {22032 ps} {590476 ps}
