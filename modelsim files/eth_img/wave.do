onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -expand -label sim:/tb_eth_img_rec/Group1 -group {Region: sim:/tb_eth_img_rec} /tb_eth_img_rec/eth_rx_clk
add wave -noupdate -expand -label sim:/tb_eth_img_rec/Group1 -group {Region: sim:/tb_eth_img_rec} /tb_eth_img_rec/rstn
add wave -noupdate -expand -label sim:/tb_eth_img_rec/Group1 -group {Region: sim:/tb_eth_img_rec} /tb_eth_img_rec/udp_date_rcev
add wave -noupdate -expand -label sim:/tb_eth_img_rec/Group1 -group {Region: sim:/tb_eth_img_rec} /tb_eth_img_rec/udp_date_en
add wave -noupdate -expand -label sim:/tb_eth_img_rec/Group1 -group {Region: sim:/tb_eth_img_rec} /tb_eth_img_rec/img_data_en
add wave -noupdate -expand -label sim:/tb_eth_img_rec/Group1 -group {Region: sim:/tb_eth_img_rec} /tb_eth_img_rec/img_data_vs
add wave -noupdate -expand -label sim:/tb_eth_img_rec/Group1 -group {Region: sim:/tb_eth_img_rec} /tb_eth_img_rec/img_data
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {315 ps} 0}
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
WaveRestoreZoom {346566 ps} {1651142 ps}
