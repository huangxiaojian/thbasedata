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
//  Class PROFILE_MANAGER
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef PROFILE_MANAGER_H
#define PROFILE_MANAGER_H

#include <windows.h>
#include <string>
#include <map>
#include <vector>

typedef std::map<std::string, std::vector<DWORD>>	TimeStorer;

class ProfileManager
{
public:
	static ProfileManager& GetInstance()
	{
		static ProfileManager profileManagerInst;
		return profileManagerInst;
	}
	void StoreSample(std::string name, DWORD elapseTime)
	{
		TimeStorer::iterator iter = dataList.find(name);
		if (iter == dataList.end())//not in the list
		{
			dataList[name] = std::vector<DWORD>(1, elapseTime);
		}
		else//in the list
		{
			dataList[name].push_back(elapseTime);
		}
	}
	void Add(TimeStorer& r)
	{
		for (TimeStorer::iterator riter = r.begin(); riter != r.end(); ++riter)
		{
			TimeStorer::iterator iter = dataList.find(riter->first);
			if (iter == dataList.end())
			{
				dataList[riter->first] = riter->second;
			}
			else
			{
				for each (DWORD t in riter->second)
				{
					iter->second.push_back(t);
				}
			}
		}
	}
	TimeStorer GetData(){ return dataList; }
private:
	ProfileManager(){}
	TimeStorer dataList;
};

#endif //PROFILE_MANAGER_H