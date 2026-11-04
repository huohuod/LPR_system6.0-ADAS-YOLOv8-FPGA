onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/clk
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/y_in
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/u_in
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/v_in
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/vs_in
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/hs_in
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/de_in
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/r_out
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/g_out
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/b_out
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/vs_out
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/hs_out
add wave -noupdate -expand -label sim:/tb_yuv2rgb/Group1 -group {Region: sim:/tb_yuv2rgb} /tb_yuv2rgb/de_out
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {426834 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 150
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 1
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
WaveRestoreZoom {0 ps} {1304576 ps}
