if(CHOLMOD_INCLUDE_DIR OR CSPARSE_INCLUDE_DIR)
  add_executable(tests test.cpp)
  target_link_libraries(tests core types_icp types_sba)
  set_target_properties(tests PROPERTIES OUTPUT_NAME tests${EXE_POSTFIX})
endif()

# Link CHOLMOD if available, CSparse otherwise
if(CHOLMOD_FOUND)
  include_directories(${CHOLMOD_INCLUDE_DIR})
  target_link_libraries(tests solver_cholmod)
elseif(CSPARSE_FOUND)
  include_directories(${CSPARSE_INCLUDE_DIR})
  target_link_libraries(tests solver_csparse)
endif()
