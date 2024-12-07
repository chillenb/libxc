link_libraries(xc)
add_executable(xc-consistency xc-consistency.c)
add_executable(xc-get_data xc-get_data.c)

if(ENABLE_CUDA)
  set_source_files_properties(xc-consistency.c xc-get_data.c PROPERTIES LANGUAGE CUDA)
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/workdir)
set(_sdirs lda_x lda_c lda_xc lda_k gga_x gga_c gga_xc gga_k mgga_x mgga_c mgga_xc mgga_k hyb_gga_x hyb_gga_xc hyb_mgga_x hyb_mgga_xc)
foreach(_sdir ${_sdirs})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/workdir/${_sdir})
    file(GLOB _bz2s RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "regression/${_sdir}/*.bz2")
    foreach(_bz2 ${_bz2s})
        # get test name
        get_filename_component(_file ${_bz2} NAME)
        string(REPLACE "." ";" _list ${_file})
        separate_arguments(_list)
        list(GET _list 0 func)
        list(GET _list 1 system)
        list(GET _list 2 spin)
        list(GET _list 3 order)
        if(${spin} STREQUAL "unpol")
            set(nspin 1)
    	else()
            set(nspin 2)
        endif()
	if(${order} EQUAL 0)
	    set(tol "5e-8")
        elseif(${order} EQUAL 1)
            set(tol "5e-5")
        elseif(${order} EQUAL 2)
            set(tol "5e-4")
	endif()
        set(refname "${func}.${system}.${spin}.${order}")

        # extract reference values
        execute_process(COMMAND bunzip2 -c "${CMAKE_CURRENT_SOURCE_DIR}/regression/${_sdir}/${refname}.bz2"
                    	OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/workdir/${_sdir}/${refname}_ref")
    endforeach()
endforeach()

# Warning: I don't know how to make a test fail, so proper capture of failure not confirmed
