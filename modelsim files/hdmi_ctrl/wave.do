onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -expand -label sim:/tb_hdmi_ctrl/Group1 -group {Region: sim:/tb_hdmi_ctrl} /tb_hdmi_ctrl/clk
add wave -noupdate -expand -label sim:/tb_hdmi_ctrl/Group1 -group {Region: sim:/tb_hdmi_ctrl} /tb_hdmi_ctrl/rst_n
add wave -noupdate -expand -label sim:/tb_hdmi_ctrl/Group1 -group {Region: sim:/tb_hdmi_ctrl} /tb_hdmi_ctrl/init_over
add wave -noupdate -expand -label sim:/tb_hdmi_ctrl/Group1 -group {Region: sim:/tb_hdmi_ctrl} /tb_hdmi_ctrl/iic_tx_scl
add wave -noupdate -expand -label sim:/tb_hdmi_ctrl/Group1 -group {Region: sim:/tb_hdmi_ctrl} /tb_hdmi_ctrl/iic_tx_sda
add wave -noupdate -expand -label sim:/tb_hdmi_ctrl/Group1 -group {Region: sim:/tb_hdmi_ctrl} /tb_hdmi_ctrl/iic_scl
add wave -noupdate -expand -label sim:/tb_hdmi_ctrl/Group1 -group {Region: sim:/tb_hdmi_ctrl} /tb_hdmi_ctrl/iic_sda
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {393902 ps} 0}
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
WaveRestoreZoom {0 ps} {2048 ns}
