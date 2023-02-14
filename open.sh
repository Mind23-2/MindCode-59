source /root/archiconda3/bin/activate mtcnn
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2
export RANK_SIZE=1

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/add-ons/:${LOCAL_ASCEND}/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
#wks demo
