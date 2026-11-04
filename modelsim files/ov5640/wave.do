onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -expand -label sim:/tb_ov5640_power_on_delay/Group1 -group {Region: sim:/tb_ov5640_power_on_delay} /tb_ov5640_power_on_delay/clk_50M
add wave -noupdate -expand -label sim:/tb_ov5640_power_on_delay/Group1 -group {Region: sim:/tb_ov5640_power_on_delay} /tb_ov5640_power_on_delay/reset_n
add wave -noupdate -expand -label sim:/tb_ov5640_power_on_delay/Group1 -group {Region: sim:/tb_ov5640_power_on_delay} /tb_ov5640_power_on_delay/camera1_rstn
add wave -noupdate -expand -label sim:/tb_ov5640_power_on_delay/Group1 -group {Region: sim:/tb_ov5640_power_on_delay} /tb_ov5640_power_on_delay/camera2_rstn
add wave -noupdate -expand -label sim:/tb_ov5640_power_on_delay/Group1 -group {Region: sim:/tb_ov5640_power_on_delay} /tb_ov5640_power_on_delay/camera_pwnd
add wave -noupdate -expand -label sim:/tb_ov5640_power_on_delay/Group1 -group {Region: sim:/tb_ov5640_power_on_delay} /tb_ov5640_power_on_delay/initial_en
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {997679 ps} 0}
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
WaveRestoreZoom {0 ps} {2609152 ps}
