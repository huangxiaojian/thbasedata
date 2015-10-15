/////////////////////////////////////////////////////////////////////////////////
///  Copyright (C), 2015-?, Xiaojian Huang
///  \file  Scalar.h
///  \brief Scalar module in SIMD library
///  \details Scalar related functions and SSE and AVX instruction micro
///  \author Xiaojian Huang
///   \version 1.0
///   \date    2015.5
///   \History: 
///                
///      Xiaojian Huang   2015.5     1.0     Create this file 
///  \warning	All rights reserved.
///  Redistribution and use in source and binary forms, with or without
///  modification, are permitted provided that the following conditions
///  are met:
///     1. Redistributions of source code must retain the above copyright
///        notice, this list of conditions and the following disclaimer.
///     2. Redistributions in binary form must reproduce the above copyright
///        notice, this list of conditions and the following disclaimer in the
///        documentation and/or other materials provided with the distribution.
///     3. The names of its contributors may not be used to endorse or promote
///        products derived from this software without specific prior written
///        permission.
///  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
///  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
///  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
///  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
///  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
///  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
///  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
///  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
///  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
///	 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
///	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////////
#ifndef TH_SCALAR_H
#define TH_SCALAR_H

#include <math.h>
#include <stdlib.h>//size_t for MSVC 6.0
#include <float.h>
#include <intrin.h>
#include <iostream>
#include <sstream>
#include <string>

#define TH_VERSION 1

inline int GetProVersion(){ return TH_VERSION; }

#define USE_SSE

//#define USE_DOUBLE_PRECISION

#define TH_UNIT 4

#ifdef USE_DOUBLE_PRECISION
#define TH_EPSILON			DBL_EPSILON
#define TH_INFINITY			DBL_MAX
#else
#define TH_EPSILON			FLT_EPSILON
#define TH_INFINITY			FLT_MAX
#endif

#define SIMD_SHUFFLE_MASK(x,y,z,w)	((w)<<6 | (z)<<4 | (y)<<2 | (x))

#ifdef USE_DOUBLE_PRECISION
typedef double Scalar;
typedef __m256d Scalar4;
#define ATTRIBUTE_ALIGNED(a)	__declspec(align(32)) a

#define SIMD_ADD(a, b)		_mm256_add_pd((a), (b))
#define SIMD_SUB(a, b)		_mm256_sub_pd((a), (b))
#define SIMD_MUL(a, b)		_mm256_mul_pd((a), (b))
#define SIMD_DIV(a, b)		_mm256_div_pd((a), (b))
#define SIMD_XOR(a, b)		_mm256_xor_pd((a), (b))
#define SIMD_MAX(a, b)		_mm256_max_pd((a), (b))
#define SIMD_MIN(a, b)		_mm256_min_pd((a), (b))
#define SIMD_AND(a, b)		_mm256_and_pd((a), (b))
#define SIMD_EQUAL(a, b)	_mm256_cmp_pd((a), (b), 0)

#define SIMD_SHUFFLE(a, _mask)	_mm256_permute4x64_pd((a), (_mask))
#define SIMD_V(v, i)		v.m256d_f64[i]
#define SIMD_V_SUM(v)		(v.m256d_f64[0]+v.m256d_f64[1]+v.m256d_f64[2])
#define SIMD_V_SUM4(v)		(v.m256d_f64[0]+v.m256d_f64[1]+v.m256d_f64[2]+v.m256d_f64[3])
#define SIMD_SET(a,b,c,d)	_mm256_set_pd((d),(c),(b),(a))
#define SIMD_TO_VEC(a,b,c)	_mm256_set_pd(0,(c),(b),(a))

#define SIMDCASTFTOI(a)		_mm256_castpd_si256(a)
#define SIMDFFF0MASK		_mm256_castsi256_pd(_mm256_set_epi64x(0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF))
#define SIMDZEROMASK		_mm256_set_pd(-0.0, -0.0, -0.0, -0.0)
#define SIMDABSMASK			_mm256_castsi256_pd(_mm256_set_epi64x(0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF))

#define SIMD_MOVEMASK(a)	_mm256_movemask_pd((a))

#define SIMDV1000				_mm256_set_pd(0.0,0.0,0.0,1.0)
#define SIMDV0100				_mm256_set_pd(0.0,0.0,1.0,0.0)
#define SIMDV0010				_mm256_set_pd(0.0,1.0,0.0,0.0)
#else
typedef float Scalar;
typedef __m128 Scalar4;
#define ATTRIBUTE_ALIGNED(a)	__declspec(align(16)) a

#define SIMD_ADD(a, b)		_mm_add_ps((a), (b))
#define SIMD_SUB(a, b)		_mm_sub_ps((a), (b))
#define SIMD_MUL(a, b)		_mm_mul_ps((a), (b))
#define SIMD_DIV(a, b)		_mm_div_ps((a), (b))
#define SIMD_XOR(a, b)		_mm_xor_ps((a), (b))
#define SIMD_MAX(a, b)		_mm_max_ps((a), (b))
#define SIMD_MIN(a, b)		_mm_min_ps((a), (b))
#define SIMD_AND(a, b)		_mm_and_ps((a), (b))
#define SIMD_EQUAL(a, b)		_mm_cmpeq_ps((a), (b))

#define SIMD_SHUFFLE(a, _mask)	_mm_shuffle_ps((a), (a), (_mask))
#define SIMD_V(v, i)		v.m128_f32[i]
#define SIMD_V_SUM(v)		(v.m128_f32[0]+v.m128_f32[1]+v.m128_f32[2])
#define SIMD_V_SUM4(v)		(v.m128_f32[0]+v.m128_f32[1]+v.m128_f32[2]+v.m128_f32[3])
#define SIMD_SET(a,b,c,d)	_mm_set_ps((d),(c),(b),(a))
#define SIMD_TO_VEC(a,b,c)	_mm_set_ps(0,(c),(b),(a))

#define SIMDCASTFTOI(a)		_mm_castps_si128((a))
#define SIMDFFF0MASK		_mm_castsi128_ps(_mm_set_epi32(0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF))
#define SIMDZEROMASK		_mm_set_ps(-0.0f, -0.0f, -0.0f, -0.0f)
#define SIMDABSMASK			_mm_castsi128_ps(_mm_set_epi32(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF))

#define SIMD_MOVEMASK(a)		_mm_movemask_ps((a))

#define SIMDV1000				_mm_set_ps(0.0f,0.0f,0.0f,1.0f)
#define SIMDV0100				_mm_set_ps(0.0f,0.0f,1.0f,0.0f)
#define SIMDV0010				_mm_set_ps(0.0f,1.0f,0.0f,0.0f)

#endif

#define SIMD_SPLAT(a, i)	SIMD_SHUFFLE((a), SIMD_SHUFFLE_MASK((i),(i),(i),(i)))
#define SIMD_SPLAT3(a, i)	SIMD_SHUFFLE((a), SIMD_SHUFFLE_MASK((i),(i),(i),3))

#define TH_FORCE_INLINE		__forceinline

#define TH_PI				Scalar(3.1415926535897932384626433832795029)
#define TH_2_PI				(Scalar(2.0) * TH_PI)
#define TH_HALF_PI			(TH_PI * Scalar(0.5))
#define TH_RADS_PER_DEG		(TH_2_PI / Scalar(360.0))
#define TH_DEGS_PER_RAD		(Scalar(360.0) / TH_2_PI)
#define TH_SQRT12			Scalar(0.7071067811865475244008443621048490)

#define TH_RECIP_SQRT(x)	((Scalar)(Scalar(1.0)/sqrt(Scalar(x))))		/* reciprocal square root */
#define TH_RECIP(x)			(Scalar(1.0)/Scalar(x))

template<typename T> 
TH_FORCE_INLINE void Swap(T& a, T& b)
{
	T tmp = a;
	a = b;
	b = tmp;
}

template <class T>
TH_FORCE_INLINE const T& thMin(const T& a, const T& b)
{
	return a < b ? a : b;
}

template <class T>
TH_FORCE_INLINE const T& thMax(const T& a, const T& b)
{
	return  a > b ? a : b;
}

template <class T>
TH_FORCE_INLINE void thSetMin(T& a, const T& b)
{
	if (b < a)
	{
		a = b;
	}
}

template <class T>
TH_FORCE_INLINE void thSetMax(T& a, const T& b)
{
	if (a < b)
	{
		a = b;
	}
}

#ifdef USE_DOUBLE_PRECISION

TH_FORCE_INLINE Scalar thSqrt(Scalar x) { return sqrt(x); }
TH_FORCE_INLINE Scalar thFabs(Scalar x) { return fabs(x); }
TH_FORCE_INLINE Scalar thCos(Scalar x) { return cos(x); }
TH_FORCE_INLINE Scalar thSin(Scalar x) { return sin(x); }
TH_FORCE_INLINE Scalar thTan(Scalar x) { return tan(x); }
TH_FORCE_INLINE Scalar thAcos(Scalar x) { if (x<Scalar(-1))	x = Scalar(-1); if (x>Scalar(1))	x = Scalar(1); return acos(x); }
TH_FORCE_INLINE Scalar thAsin(Scalar x) { if (x<Scalar(-1))	x = Scalar(-1); if (x>Scalar(1))	x = Scalar(1); return asin(x); }
TH_FORCE_INLINE Scalar thAtan(Scalar x) { return atan(x); }
TH_FORCE_INLINE Scalar thAtan2(Scalar x, Scalar y) { return atan2(x, y); }
TH_FORCE_INLINE Scalar thExp(Scalar x) { return exp(x); }
TH_FORCE_INLINE Scalar thLog(Scalar x) { return log(x); }
TH_FORCE_INLINE Scalar thPow(Scalar x, Scalar y) { return pow(x, y); }
TH_FORCE_INLINE Scalar thFmod(Scalar x, Scalar y) { return fmod(x, y); }

#else

TH_FORCE_INLINE Scalar thSqrt(Scalar y){ return sqrtf(y); }

TH_FORCE_INLINE Scalar thFastSqrt(Scalar y)
{
	double x, z, tempf;
	unsigned long *tfptr = ((unsigned long *)&tempf) + 1;

	tempf = y;
	*tfptr = (0xbfcdd90a - *tfptr) >> 1; /* estimate of 1/sqrt(y) */
	x = tempf;
	z = y*Scalar(0.5);
	x = (Scalar(1.5)*x) - (x*x)*(x*z);         /* iteration formula     */
	x = (Scalar(1.5)*x) - (x*x)*(x*z);
	x = (Scalar(1.5)*x) - (x*x)*(x*z);
	x = (Scalar(1.5)*x) - (x*x)*(x*z);
	x = (Scalar(1.5)*x) - (x*x)*(x*z);
	return x*y;
}

TH_FORCE_INLINE Scalar thFabs(Scalar x) { return fabsf(x); }
TH_FORCE_INLINE Scalar thCos(Scalar x) { return cosf(x); }
TH_FORCE_INLINE Scalar thSin(Scalar x) { return sinf(x); }
TH_FORCE_INLINE Scalar thTan(Scalar x) { return tanf(x); }
TH_FORCE_INLINE Scalar thAcos(Scalar x) {
	if (x<Scalar(-1))
		x = Scalar(-1);
	if (x>Scalar(1))
		x = Scalar(1);
	return acosf(x);
}
TH_FORCE_INLINE Scalar thAsin(Scalar x) {
	if (x<Scalar(-1))
		x = Scalar(-1);
	if (x>Scalar(1))
		x = Scalar(1);
	return asinf(x);
}
TH_FORCE_INLINE Scalar thAtan(Scalar x) { return atanf(x); }
TH_FORCE_INLINE Scalar thAtan2(Scalar x, Scalar y) { return atan2f(x, y); }
TH_FORCE_INLINE Scalar thExp(Scalar x) { return expf(x); }
TH_FORCE_INLINE Scalar thLog(Scalar x) { return logf(x); }
TH_FORCE_INLINE Scalar thPow(Scalar x, Scalar y) { return powf(x, y); }
TH_FORCE_INLINE Scalar thFmod(Scalar x, Scalar y) { return fmodf(x, y); }

#endif

// returns normalized value in range [-TH_PI, TH_PI]
TH_FORCE_INLINE Scalar thNormalizeAngle(Scalar angleInRadians)
{
	angleInRadians = thFmod(angleInRadians, TH_2_PI);
	if (angleInRadians < -TH_PI)
	{
		return angleInRadians + TH_2_PI;
	}
	else if (angleInRadians > TH_PI)
	{
		return angleInRadians - TH_2_PI;
	}
	else
	{
		return angleInRadians;
	}
}

TH_FORCE_INLINE Scalar thAtan2Fast(Scalar y, Scalar x)
{
	Scalar coeff_1 = TH_PI / 4.0f;
	Scalar coeff_2 = 3.0f * coeff_1;
	Scalar abs_y = thFabs(y);
	Scalar angle;
	if (x >= 0.0f) {
		Scalar r = (x - abs_y) / (x + abs_y);
		angle = coeff_1 - coeff_1 * r;
	}
	else {
		Scalar r = (x + abs_y) / (abs_y - x);
		angle = coeff_2 - coeff_1 * r;
	}
	return (y < 0.0f) ? -angle : angle;
}

TH_FORCE_INLINE bool thFuzzyZero(Scalar x) { return thFabs(x) < TH_EPSILON; }
TH_FORCE_INLINE bool thEqual(Scalar a, Scalar eps) { return (((a) <= eps) && !((a) < -eps)); }
TH_FORCE_INLINE bool thGreaterEqual(Scalar a, Scalar eps) { return (!((a) <= eps)); }
TH_FORCE_INLINE bool thIsNegative(Scalar x) { return x < Scalar(0.0) ? true : false; }

//Degree -> Radian
TH_FORCE_INLINE Scalar thRadians(Scalar x) { return x * TH_RADS_PER_DEG; }

//Radian -> Degree
TH_FORCE_INLINE Scalar thDegrees(Scalar x) { return x * TH_DEGS_PER_RAD; }

#endif //TH_SCALAR_H