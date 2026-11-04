`timescale 1ns / 1ps

module tb_eth_img_rec;

// 输入
reg                         eth_rx_clk;
reg                         rstn;
reg [31:0]                  udp_date_rcev;
reg                         udp_date_en;

// 输出
wire                        img_data_en;
wire                        img_data_vs;
wire [15:0]                 img_data;

// 例化待测模块
eth_img_rec 
#(
    .PIXEL_WIDTH    (32),
    .VIDEO_LENGTH   (960),
    .VIDEO_HIGTH    (540)
)
u_eth_img_rec (
    .eth_rx_clk     (eth_rx_clk),
    .rstn           (rstn),
    .udp_date_rcev  (udp_date_rcev),
    .udp_date_en    (udp_date_en),
    
    .img_data_en    (img_data_en),
    .img_data_vs     (img_data_vs),
    .img_data       (img_data)
);

// 50MHz 时钟
always #10 eth_rx_clk = ~eth_rx_clk;

// 激励：模拟以太网图像接收
initial begin
    eth_rx_clk    = 0;
    rstn          = 0;
    udp_date_rcev = 0;
    udp_date_en   = 0;
    
    #100;
    rstn = 1;   // 释放复位
    
    #50;
    // 1. 发送帧头 32'hf0_5a_a5_0f
    udp_date_en   = 1;
    udp_date_rcev = 32'hf0_5a_a5_0f;
    #20;
    
    // 2. 发送分辨率 960x540
    udp_date_en   = 1;
    udp_date_rcev = {16'd960, 16'd540};
    #20;
    
    // 3. 发送图像数据（测试数据）
    udp_date_en   = 1;
    udp_date_rcev = 32'h12345678;
    #20;
    
    udp_date_en   = 1;
    udp_date_rcev = 32'h87654321;
    #20;
    
    udp_date_en   = 0;
    
    #1000;
    $stop;
end

endmodule