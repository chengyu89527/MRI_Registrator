/*  sweopts.cc
    Bryan Guillaume & Tom Nichols
    Copyright (C) 2019 University of Oxford  */

#ifndef ____sweopts__
#define ____sweopts__
#define WANT_STREAM
#define WANT_MATH

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "sweopts.h"

#endif /* defined(____sweopts__) */

using namespace Utilities;

namespace SWE {
	
	sweopts* sweopts::gopt = NULL;
	
	void sweopts::parse_command_line(int argc, char** argv, Log& logger)
	{
		
		//Do the parsing;
		try{
			for(int a = options.parse_command_line(argc, argv); a < argc; a++);
			
			if(help.value() || ! options.check_compulsory_arguments())
			{
				options.usage();
				exit(2);
			}
		}
		catch(X_OptionError& e){
			cerr << e.what() << endl;
			cerr << "try: swe --help" << endl;
			exit(1);
		}
	}
}

