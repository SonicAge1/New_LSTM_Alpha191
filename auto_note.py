import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header
import itchat

def send_email_with_attachment(attachment_path, subject, zw):

    smtp_server = "smtp.qq.com"
    smtp_port = 465  # SMTP服务端口号

    # 邮箱账号和授权码
    username = "1966488323@qq.com"  # 邮箱账号
    auth_code = "mjnrrmzfhmkncfge"  # 邮箱授权码
    sender_email = username
    attachment_path = attachment_path  # 图片文件的路径

    receiver_email="1966488323@qq.com"
    body=zw

    # 创建邮件对象
    message = MIMEMultipart()
    message['From'] = Header(sender_email)
    message['To'] = Header(receiver_email)
    message['Subject'] = Header(subject)

    # 邮件正文内容
    message.attach(MIMEText(body, 'plain', 'utf-8'))

    # 添加附件
    with open(attachment_path, 'rb') as attachment_file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment_file.read())
        encoders.encode_base64(part)  # 对附件进行base64编码
        part.add_header(
            'Content-Disposition',
            f'attachment; filename={attachment_path}',
        )
        message.attach(part)

    # 使用smtplib发送邮件
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(username, auth_code)
            server.sendmail(sender_email, receiver_email, message.as_string())
            print("邮件发送成功")
    except Exception as e:
        print(f"邮件发送失败: {e}")


def send_wechat_message(to_user, message):
    users = itchat.search_friends(name=to_user)  # 搜索好友，name为微信好友的备注名称
    if users:
        user = users[0]["UserName"]
        itchat.send(msg=message, toUserName=user)  # 发送消息
        print(f"消息已发送给 {to_user}")
    else:
        print(f"未找到用户: {to_user}")
    itchat.logout()  # 登出微信

# # 使用示例：替换'好友备注'和'你想发送的消息'
# itchat.auto_login(hotReload=True)  # 登录微信，hotReload=True参数表示之后自动登录
# send_wechat_message('妈', 'lll')

