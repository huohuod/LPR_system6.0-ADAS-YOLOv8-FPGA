onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_ACLK
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_ARESETN
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/DDR_INIT_DONE
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_AWADDR
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_AWVALID
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_AWREADY
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_WLAST
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_WREADY
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_ARADDR
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_ARVALID
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_ARREADY
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_RLAST
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/M_AXI_RVALID
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/vs_in
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/vs_out
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/wfifo_rd_water_level
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/wfifo_rd_req
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/wfifo_pre_rd_req
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/rfifo_wr_water_level
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/rfifo_wr_req
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/r_fram_done
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/wr_addr_min
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/wr_addr_max
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/r_wr_rst
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/r_rd_rst
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/w_fifo_state
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/r_fifo_state
add wave -noupdate -expand -label sim:/tb_AXI_FULL_M/Group1 -group {Region: sim:/tb_AXI_FULL_M} /tb_AXI_FULL_M/wr_addr_cnt
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {0 ps} 0}
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
WaveRestoreZoom {0 ps} {1024 ns}
