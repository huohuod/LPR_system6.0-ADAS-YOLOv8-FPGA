`timescale 1ns / 1ps

module tb_hdmi_ctrl;

// 时钟 & 复位
reg         clk;
reg         rst_n;

// 输出
wire        init_over;
wire        iic_tx_scl;
wire        iic_tx_sda;
wire        iic_scl;
wire        iic_sda;

// 例化待测模块
hdmi_ctrl u_hdmi_ctrl (
    .clk          (clk),
    .rst_n        (rst_n),
    
    .init_over    (init_over),
    .iic_tx_scl   (iic_tx_scl),
    .iic_tx_sda   (iic_tx_sda),
    .iic_scl      (iic_scl),
    .iic_sda      (iic_sda)
);

// 10MHz 时钟（和模块里配置一致）
always #50 clk = ~clk;

// 复位激励
initial begin
    clk   = 0;
    rst_n = 0;
    
    #100;
    rst_n = 1;
    
    #200000;
    $stop;
end

endmodule