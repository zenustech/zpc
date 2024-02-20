
if (EXISTS ${OUTPUT})
	foreach(obj ${OBJECTS})
		if (EXISTS ${obj})
			execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${obj} ${OUTPUT})
			message("copied [${obj}] to [${OUTPUT}]")
		endif()
	endforeach()
endif()