///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2015, Xiaojian Huang
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////////////////
//  Class AUTO_PROFILE
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef AUTO_PROFILE_H
#define AUTO_PROFILE_H

#include "ProfileManager.h"
#include <iostream>

struct AutoProfile
{
	AutoProfile(const char* filename)
	{
		m_name = filename;
		m_startTime = GetTickCount();
	}

	~AutoProfile()
	{
		DWORD elapseTime = GetTickCount() - m_startTime;
		std::cout << m_name << " cost " << elapseTime << "ms" << std::endl;
		ProfileManager::GetInstance().StoreSample(m_name, elapseTime);
	}

	const char* m_name;
	DWORD	 m_startTime;
};

#define AUTO_PROFILE_BEGIN(name) {AutoProfile auto_profile_analyser(name);
#define AUTO_PROFILE_END	}

#endif //AUTO_PROFILE_H