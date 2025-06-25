#!/bin/bash
# Run all tests for PhotonicFusion SDXL

echo "🧪 PhotonicFusion SDXL - 运行所有测试"
echo "=================================="

# 设置颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# 测试结果计数
PASSED=0
FAILED=0
TOTAL=0

# 运行测试并检查结果
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "\n${YELLOW}运行测试: ${test_name}${NC}"
    echo "命令: $test_command"
    echo "----------------------------------------"
    
    if eval $test_command; then
        echo -e "${GREEN}✅ 测试通过: ${test_name}${NC}"
        PASSED=$((PASSED+1))
    else
        echo -e "${RED}❌ 测试失败: ${test_name}${NC}"
        FAILED=$((FAILED+1))
    fi
    
    TOTAL=$((TOTAL+1))
    echo "----------------------------------------"
}

# 检查 Python 环境
echo -e "\n${YELLOW}检查 Python 环境${NC}"
python --version
pip --version

# 检查依赖
echo -e "\n${YELLOW}检查关键依赖${NC}"
pip list | grep -E "torch|diffusers|transformers|protobuf"

# 运行所有测试
echo -e "\n${YELLOW}开始运行测试...${NC}"

# 1. device_map 修复测试
run_test "device_map 修复" "python test_fix.py"

# 2. protobuf 修复测试
run_test "protobuf 修复" "python test_protobuf_fix.py"

# 3. 导入测试
run_test "导入测试" "python -c 'import handler; print(\"Handler 导入成功\")'"

# 4. 模型加载测试
run_test "模型加载测试" "python -c 'from handler import load_model; print(\"模型加载函数导入成功\")'"

# 5. 语法检查
if command -v black &> /dev/null; then
    run_test "代码格式检查" "black --check ."
fi

# 显示测试结果摘要
echo -e "\n${YELLOW}测试结果摘要${NC}"
echo "----------------------------------------"
echo -e "总测试数: ${TOTAL}"
echo -e "${GREEN}通过: ${PASSED}${NC}"
echo -e "${RED}失败: ${FAILED}${NC}"
echo "----------------------------------------"

# 显示下一步操作
echo -e "\n${YELLOW}下一步操作${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}所有测试通过! 可以部署到 RunPod.${NC}"
    echo "1. 构建 Docker 镜像: ./build_fixed_docker.sh"
    echo "2. 将镜像推送到您的容器仓库"
    echo "3. 在 RunPod 上更新端点配置"
    echo "4. 使用 test_api_request.json 测试 API"
else
    echo -e "${RED}有测试失败，请修复问题后再部署。${NC}"
fi

# 退出状态基于测试结果
if [ $FAILED -eq 0 ]; then
    exit 0
else
    exit 1
fi 