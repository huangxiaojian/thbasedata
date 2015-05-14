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
//  Class QUATERNION
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef TH_QUATERNIONS_H
#define TH_QUATERNIONS_H

#include "Scalar.h"
#include "Vector.h"
#include "Matrix.h"
#include <iostream>

ATTRIBUTE_ALIGNED(class) Quaternion
{
public:
	union {
		Scalar4	m_vec;
		Scalar m_scalar[4];
	};

	inline Scalar4 get() const { return m_vec; }
	inline void set(Scalar4& v){ m_vec = v; }
	inline void	setValue(const Scalar& _x, const Scalar& _y, const Scalar& _z, const Scalar& _w)
	{
		m_scalar[0] = _x;
		m_scalar[1] = _y;
		m_scalar[2] = _z;
		m_scalar[3] = _w;
	}

	inline explicit Quaternion()
	{
#ifdef USE_SSE
		m_vec = SIMD_SET(1.0, 1.0, 1.0, 1.0);
#else
		m_scalar[0] = 1.0;
		m_scalar[1] = 1.0;
		m_scalar[2] = 1.0;
		m_scalar[3] = 1.0;
#endif
	}

	inline explicit Quaternion(const Scalar x, const Scalar y, const Scalar z, const Scalar w)
	{
#ifdef USE_SSE
		m_vec = SIMD_SET(x, y, z, w);
#else
		m_scalar[0] = x;
		m_scalar[1] = y;
		m_scalar[2] = z;
		m_scalar[3] = w;
#endif
	}

	inline Quaternion(const Quaternion& other){ m_vec = other.m_vec; }
	inline explicit Quaternion(const Scalar4& other){ m_vec = other; }

	/**
	*  Constructs a quaternion from a rotation matrix.
	*/
	//TODO
	/*inline explicit Quaternion(const Matrix3x4& mat)
	{
	  
	}*/
	


	/**@brief Return the negative of this quaternion
	* This simply negates each element */
	inline Quaternion operator-() const
	{
#if defined USE_SSE
		//TODO
		//return Quaternion(SIMD_XOR(m_vec, btvMzeroMask));
		const Quaternion& q2 = *this;
		return Quaternion(-q2.x(), -q2.y(), -q2.z(), -q2.w());
#else	
		const Quaternion& q2 = *this;
		return Quaternion(-q2.x(), -q2.y(), -q2.z(), -q2.w());
#endif
	}

	friend Quaternion operator+(const Quaternion& v1, const Quaternion& v2);
	friend Quaternion operator-(const Quaternion& v1, const Quaternion& v2);
	friend Quaternion operator*(const Quaternion& v1, const Quaternion& v2);
	friend Quaternion operator*(const Quaternion& v1, const Scalar& v2);
	friend Quaternion operator/(const Quaternion& v1, const Scalar& v2);

	inline Quaternion& operator=(const Quaternion& other)
	{
		m_vec = other.m_vec;
		return *this;
	}

	inline Quaternion& operator-=(const Quaternion& other)
	{
#ifdef USE_SSE
		m_vec = SIMD_SUB(m_vec, other.m_vec);
#else
		m_scalar[0] -= other.m_scalar[0];
		m_scalar[1] -= other.m_scalar[1];
		m_scalar[2] -= other.m_scalar[2];
		m_scalar[3] -= other.m_scalar[3];
#endif
		return *this;
	}

	inline Quaternion& operator+=(const Quaternion& other)
	{
#ifdef USE_SSE
		m_vec = SIMD_ADD(m_vec, other.m_vec);
#else
		m_scalar[0] += other.m_scalar[0];
		m_scalar[1] += other.m_scalar[1];
		m_scalar[2] += other.m_scalar[2];
		m_scalar[3] += other.m_scalar[3];
#endif
		return *this;
	}

	inline Quaternion& operator*=(const Quaternion& other)
	{
#ifdef USE_SSE
		//TODO
		setValue(
			m_scalar[3] * other.x() + m_scalar[0] * other.w() + m_scalar[1] * other.z() - m_scalar[2] * other.y(),
			m_scalar[3] * other.y() + m_scalar[1] * other.w() + m_scalar[2] * other.x() - m_scalar[0] * other.z(),
			m_scalar[3] * other.z() + m_scalar[2] * other.w() + m_scalar[0] * other.y() - m_scalar[1] * other.x(),
			m_scalar[3] * other.w() - m_scalar[0] * other.x() - m_scalar[1] * other.y() - m_scalar[2] * other.z());
#else
		setValue(
			m_scalar[3] * other.x() + m_scalar[0] * other.w() + m_scalar[1] * other.z() - m_scalar[2] * other.y(),
			m_scalar[3] * other.y() + m_scalar[1] * other.w() + m_scalar[2] * other.x() - m_scalar[0] * other.z(),
			m_scalar[3] * other.z() + m_scalar[2] * other.w() + m_scalar[0] * other.y() - m_scalar[1] * other.x(),
			m_scalar[3] * other.w() - m_scalar[0] * other.x() - m_scalar[1] * other.y() - m_scalar[2] * other.z());

#endif
		return *this;
	}

	inline Quaternion& operator/=(const Scalar& v)
	{
#ifdef USE_SSE
		//TODO
		Quaternion other(v, v, v, v);
		m_vec = SIMD_DIV(m_vec, other.m_vec);
#else
		m_scalar[0] /= v;
		m_scalar[1] /= v;
		m_scalar[2] /= v;
		m_scalar[3] /= v;
#endif
		return *this;
	}


	inline Quaternion& operator*=(const Scalar& v)
	{
#ifdef USE_SSE
		Quaternion other(v, v, v, v);
		m_vec = SIMD_MUL(m_vec, other.m_vec);
#else
		m_scalar[0] *= v;
		m_scalar[1] *= v;
		m_scalar[2] *= v;
		m_scalar[3] *= v;
#endif
		return *this;
	}


	//TODO
	bool operator==(const Quaternion& other) const
	{
		return ((m_scalar[3] == other.m_scalar[3]) &&
			(m_scalar[2] == other.m_scalar[2]) &&
			(m_scalar[1] == other.m_scalar[1]) &&
			(m_scalar[0] == other.m_scalar[0]));
	}

	//TODO
	bool operator!=(const Quaternion& other) const
	{
		return !(*this == other);
	}



	/**@brief Return the dot product
	* @param other: The other Quaternion in the dot product */
	inline Scalar dot(const Quaternion& other) const
	{
#ifdef USE_SSE
		Scalar4 vd = SIMD_MUL(m_vec, other.m_vec);
		return SIMD_V_SUM4(vd);
#else
		return m_scalar[0] * other.m_scalar[0] + m_scalar[1] * other.m_scalar[1] + m_scalar[2] * other.m_scalar[2] + m_scalar[3] * other.m_scalar[3];
#endif
	}

	/**@brief Return the length of the Quaternion squared */
	inline Scalar length2() const
	{
		return dot(*this);
	}

	/**@brief Return the length of the Quaternion */
	inline Scalar length() const
	{
		return thSqrt(length2());
	}

	/**@brief Return the norm (length) of the Quaternion */
	inline Scalar norm() const
	{
		return length();
	}

	//TODO
	/**@brief Normalize the quaternion
	* Such that x^2 + y^2 + z^2 +w^2 = 1 */
	Quaternion& normalize()
	{
#if defined USE_SSE
		//__m128	vd;

		//vd = _mm_mul_ps(mVec128, mVec128);

		//__m128 t = _mm_movehl_ps(vd, vd);
		//vd = _mm_add_ps(vd, t);
		//t = _mm_shuffle_ps(vd, vd, 0x55);
		//vd = _mm_add_ss(vd, t);

		//vd = _mm_sqrt_ss(vd);
		//vd = _mm_div_ss(vOnes, vd);
		//vd = bt_pshufd_ps(vd, 0); // splat
		//mVec128 = _mm_mul_ps(mVec128, vd);

	    return *this /= length();
#else    
		return *this /= length();
#endif
	}

	//TODO
	/**@brief Return a normalized version of this Quaternion */
	Quaternion normalized() const
	{   //should be modified
		Scalar l = length();
		l = l <= TH_EPSILON ? 1 : l;
		return *this / length();
	}

	/**@brief Return the ***half*** angle between this quaternion and the other
	* @param q The other quaternion */
	Scalar angle(const Quaternion& other) const
	{
		Scalar s = thSqrt(length2() * other.length2());
		//btAssert(s != Scalar(0.0));
		return thAcos(dot(other) / s);
	}

	/**@brief Return the angle between this quaternion and the other along the shortest path
	* @param q The other quaternion */
	Scalar angleShortestPath(const Quaternion& q) const
	{
		Scalar s = thSqrt(length2() * q.length2());
		//Assert(s != Scalar(0.0));
		if (dot(q) < 0) // Take care of long angle case see http://en.wikipedia.org/wiki/Slerp
			return thAcos(dot(-q) / s) * Scalar(2.0);
		else
			return thAcos(dot(q) / s) * Scalar(2.0);
	}

	/**@brief Return the quaternion which is the result of Spherical Linear Interpolation between this and the other quaternion
	* @param q The other quaternion to interpolate with
	* @param t The ratio between this and q to interpolate.  If t = 0 the result is this, if t=1 the result is q.
	* Slerp interpolates assuming constant velocity.  */
	Quaternion slerp(const Quaternion& q, const Scalar& t) const
	{
		Scalar magnitude = thSqrt(length2() * q.length2());
		//btAssert(magnitude > Scalar(0));

		Scalar product = dot(q) / magnitude;
		if (thFabs(product) < Scalar(1))
		{
			// Take care of long angle case see http://en.wikipedia.org/wiki/Slerp
			const Scalar sign = (product < 0) ? Scalar(-1) : Scalar(1);

			const Scalar theta = thAcos(sign * product);
			const Scalar s1 = thSin(sign * t * theta);
			const Scalar d = Scalar(1.0) / thSin(theta);
			const Scalar s0 = thSin((Scalar(1.0) - t) * theta);

			return Quaternion(
				(m_scalar[0] * s0 + q.x() * s1) * d,
				(m_scalar[1] * s0 + q.y() * s1) * d,
				(m_scalar[2] * s0 + q.z() * s1) * d,
				(m_scalar[3] * s0 + q.w() * s1) * d);
		}
		else
		{
			return *this;
		}
	}


	/**@brief Set the quaternion using Euler angles
	* @param yaw Angle around Y
	* @param pitch Angle around X
	* @param roll Angle around Z */
	void setEuler(const Scalar& yaw, const Scalar& pitch, const Scalar& roll)
	{
		Scalar halfYaw = Scalar(yaw) * Scalar(0.5);
		Scalar halfPitch = Scalar(pitch) * Scalar(0.5);
		Scalar halfRoll = Scalar(roll) * Scalar(0.5);
		Scalar cosYaw = thCos(halfYaw);
		Scalar sinYaw = thSin(halfYaw);
		Scalar cosPitch = thCos(halfPitch);
		Scalar sinPitch = thSin(halfPitch);
		Scalar cosRoll = thCos(halfRoll);
		Scalar sinRoll = thSin(halfRoll);
		setValue(cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw,
			cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw,
			sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw,
			cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw);
	}


	/**@brief Set the quaternion using euler angles
	* @param yaw Angle around Z
	* @param pitch Angle around Y
	* @param roll Angle around X */
	void setEulerZYX(const Scalar& yaw, const Scalar& pitch, const Scalar& roll)
	{
		Scalar halfYaw = Scalar(yaw) * Scalar(0.5);
		Scalar halfPitch = Scalar(pitch) * Scalar(0.5);
		Scalar halfRoll = Scalar(roll) * Scalar(0.5);
		Scalar cosYaw = thCos(halfYaw);
		Scalar sinYaw = thSin(halfYaw);
		Scalar cosPitch = thCos(halfPitch);
		Scalar sinPitch = thSin(halfPitch);
		Scalar cosRoll = thCos(halfRoll);
		Scalar sinRoll = thSin(halfRoll);
		setValue(sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw, //x
			cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw, //y
			cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw, //z
			cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw); //formerly yzx
	}

	/**@brief Set the rotation using axis angle notation
	* @param axis The axis around which to rotate
	* @param angle The magnitude of the rotation in Radians */
	void setRotation(const Vector4& axis, const Scalar& _angle)
	{
		Scalar d = axis.length();
		//btAssert(d != Scalar(0.0));
		Scalar s = thSin(_angle * Scalar(0.5)) / d;
		setValue(axis.x() * s, axis.y() * s, axis.z() * s,
			thCos(_angle * Scalar(0.5)));
	}

	/**@brief Return the angle of rotation represented by this quaternion */
	Scalar getAngle() const
	{
		Scalar s = Scalar(2.) * thAcos(m_scalar[3]);
		return s;
	}

	/**@brief Return the angle of rotation represented by this quaternion along the shortest path*/
	Scalar getAngleShortestPath() const
	{
		Scalar s;
		if (dot(*this) < 0)
			s = Scalar(2.) * thAcos(m_scalar[3]);
		else
			s = Scalar(2.) * thAcos(-m_scalar[3]);

		return s;
	}

	/**@brief Return the axis of the rotation represented by this quaternion */
	Vector4 getAxis() const
	{
		Scalar s_squared = 1.f - m_scalar[3] * m_scalar[3];

		if (s_squared < Scalar(10.) * TH_EPSILON) //Check for divide by zero
			return Vector4(1.0, 0.0, 0.0);  // Arbitrary
		Scalar s = 1.f / thSqrt(s_squared);
		return Vector4(m_scalar[0] * s, m_scalar[1] * s, m_scalar[2] * s);
	}

	static const Quaternion&	getIdentity()
	{
		static const Quaternion identityQuat(Scalar(0.), Scalar(0.), Scalar(0.), Scalar(1.));
		return identityQuat;
	}

	//TODO
	/**@brief Return the inverse of this quaternion */
	Quaternion inverse() const
	{
#if defined USE_SSE
		//return Quaternion(_mm_xor_ps(mVec128, vQInv));
		return Quaternion(-m_scalar[0], -m_scalar[1], -m_scalar[2], m_scalar[3]);
#else	
		return Quaternion(-m_scalar[0], -m_scalar[1], -m_scalar[2], m_scalar[3]);
#endif
	}

	//TODO
	/**@todo document this and it's use */
	inline Quaternion nearest(const Quaternion& qd) const
	{
		/*Quaternion diff, sum;
		diff = *this - qd;
		sum = *this + qd;
		if (diff.dot(diff) < sum.dot(sum))
			return qd;
		return (-qd);*/
	}



	/**@brief Return the x value */
	inline const Scalar& x() const { return m_scalar[0]; }
	/**@brief Return the y value */
	inline const Scalar& y() const { return m_scalar[1]; }
	/**@brief Return the z value */
	inline const Scalar& z() const { return m_scalar[2]; }
	/**@brief Return the w value */
	inline const Scalar& w() const { return m_scalar[3]; }

	inline void print()
	{
		std::cout << m_scalar[0] << ", " << m_scalar[1]
			<< ", " << m_scalar[2] << ", " << m_scalar[3] << std::endl;
	}

	inline std::string toString()
	{
		std::ostringstream os;
		os << "(" << m_scalar[0] << ", " << m_scalar[1]
			<< ", " << m_scalar[2] << ", " << m_scalar[3] << ")" << std::endl;
		return os.str();
	}

};  //CLASS DECLARE END







//friend function for Quaternion
//+, -, *, /, *(scalar), /(scalar)
inline Quaternion operator+(const Quaternion& v1, const Quaternion& v2)
{
#ifdef USE_SSE
	Scalar4 vec = SIMD_ADD(v1.m_vec, v2.m_vec);
	return Quaternion(vec);
#else
	return Quaternion(v1.m_scalar[0] + v2.m_scalar[0], v1.m_scalar[1] + v2.m_scalar[1],
		v1.m_scalar[2] + v2.m_scalar[2], v1.m_scalar[3] + v2.m_scalar[3]);
#endif
}

inline Quaternion operator-(const Quaternion& v1, const Quaternion& v2)
{
#ifdef USE_SSE
	Scalar4 vec = SIMD_SUB(v1.m_vec, v2.m_vec);
	return Quaternion(vec);
#else
	return Quaternion(v1.m_scalar[0] - v2.m_scalar[0], v1.m_scalar[1] - v2.m_scalar[1],
		v1.m_scalar[2] - v2.m_scalar[2], v1.m_scalar[3] - v2.m_scalar[3]);
#endif
}


//TODO
inline Quaternion operator*(const Quaternion& v1, const Quaternion& v2)
{
#ifdef USE_SSE
	return Quaternion(
		v1.w() * v2.x() + v1.x() * v2.w() + v1.y() * v2.z() - v1.z() * v2.y(),
		v1.w() * v2.y() + v1.y() * v2.w() + v1.z() * v2.x() - v1.x() * v2.z(),
		v1.w() * v2.z() + v1.z() * v2.w() + v1.x() * v2.y() - v1.y() * v2.x(),
		v1.w() * v2.w() - v1.x() * v2.x() - v1.y() * v2.y() - v1.z() * v2.z());
#else
	return Quaternion(
		v1.w() * v2.x() + v1.x() * v2.w() + v1.y() * v2.z() - v1.z() * v2.y(),
		v1.w() * v2.y() + v1.y() * v2.w() + v1.z() * v2.x() - v1.x() * v2.z(),
		v1.w() * v2.z() + v1.z() * v2.w() + v1.x() * v2.y() - v1.y() * v2.x(),
		v1.w() * v2.w() - v1.x() * v2.x() - v1.y() * v2.y() - v1.z() * v2.z());
#endif
}

//TODO
inline Quaternion operator*(const Quaternion& v1, const Vector4& w)
{
#ifdef USE_SSE
	return Quaternion(
		v1.w() * w.x() + v1.y() * w.z() - v1.z() * w.y(),
		v1.w() * w.y() + v1.z() * w.x() - v1.x() * w.z(),
		v1.w() * w.z() + v1.x() * w.y() - v1.y() * w.x(),
		-v1.x() * w.x() - v1.y() * w.y() - v1.z() * w.z());
#else
	return Quaternion(
		v1.w() * w.x() + v1.y() * w.z() - v1.z() * w.y(),
		v1.w() * w.y() + v1.z() * w.x() - v1.x() * w.z(),
		v1.w() * w.z() + v1.x() * w.y() - v1.y() * w.x(),
		-v1.x() * w.x() - v1.y() * w.y() - v1.z() * w.z()); 
#endif
}

//TODO
inline Quaternion operator*(const Vector4& w, const Quaternion& q)
{
#if defined USE_SSE
	return Quaternion(
		+w.x() * q.w() + w.y() * q.z() - w.z() * q.y(),
		+w.y() * q.w() + w.z() * q.x() - w.x() * q.z(),
		+w.z() * q.w() + w.x() * q.y() - w.y() * q.x(),
		-w.x() * q.x() - w.y() * q.y() - w.z() * q.z());
#else
	return Quaternion(
		+w.x() * q.w() + w.y() * q.z() - w.z() * q.y(),
		+w.y() * q.w() + w.z() * q.x() - w.x() * q.z(),
		+w.z() * q.w() + w.x() * q.y() - w.y() * q.x(),
		-w.x() * q.x() - w.y() * q.y() - w.z() * q.z());
#endif
}

inline Quaternion operator*(const Quaternion& v1, const Scalar& v2)
{
#ifdef USE_SSE
	Quaternion v(v2, v2, v2, v2);
	Scalar4 vec = SIMD_MUL(v1.m_vec, v.m_vec);
	return Quaternion(vec);
#else
	return Quaternion(v1.m_scalar[0] * v2, v1.m_scalar[1] * v2, v1.m_scalar[2] * v2, v1.m_scalar[3] * v2);
#endif
}

inline Quaternion operator/(const Quaternion& v1, const Scalar& v2)
{
#ifdef USE_SSE
	Quaternion v(v2, v2, v2, v2);
	Scalar4 vec = SIMD_DIV(v1.m_vec, v.m_vec);
	return Quaternion(vec);
#else
	return Quaternion(v1.m_scalar[0] / v2, v1.m_scalar[1] / v2, v1.m_scalar[2] / v2, , v1.m_scalar[3] / v2);
#endif
}












#endif //TH_QUATERNIONS_H